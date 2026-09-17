//! Turns the engine's layout registry into the Python starterpack's `ctypes` bindings.
//!
//! `#[derive(FfiMirror)]`, `#[mm_ffi_fn]` and `#[mm_ffi_handle]` each record what they
//! describe into an `inventory` registry (see `mm_engine::game::mirror` and
//! `mm_engine::ffi`). This binary links `mm-engine` as an rlib, reads all five registries,
//! and writes `core/_generated/`:
//!
//! - `bindings.py` -- the `ctypes.Structure`/`Union` mirror of every wire type, the length
//!   constants, and the `CDLL` bindings for every `extern "C"` entry point;
//! - `__init__.py` -- a re-export, so `from core._generated import GameState` works;
//! - `layout.json` -- the registry dumped verbatim, which `tools/tests/` checks the emitted
//!   `ctypes` classes against.
//!
//! # Why the generator lives here and not in `mm-engine`
//!
//! Same constraint that puts the cdylib in this crate: `cargo build`/`cargo install` of a
//! *dependency* surfaces neither its cdylib nor its binaries anywhere findable. Living in
//! the shim crate also means it is automatically built with the same features as the `.so`
//! it describes -- `default-features = false, features = ["client", "ffi"]`, which is what
//! makes the output carry the `Me`/`Other` naming competitors are given rather than the
//! engine's absolute `A`/`B`.
//!
//! # What this emits
//!
//! Both layers, in one module. The **wire layer** is layout-exact `ctypes` classes and the
//! raw entry points. The **friendly surface** `dev/ffi.md` describes -- `StateOption` as
//! `Optional[T]`, `BotArray` as a sequence, `TeamPair` indexed by `Team`, `BotState`'s class
//! accessors, data enums as tagged classes -- is emitted onto those same classes rather than
//! wrapped by a hand-written layer, because a hand-written mirror over a struct that gains a
//! field drifts, and that is the failure this whole surface exists to prevent. See "the
//! friendly surface" below for the two rules and the one declared table.
//!
//! `core/channel.py` is what is left over and genuinely hand-written: the channel loop and
//! the navigation wrappers, neither of which the registry describes.
//!
//! Output is deterministic: `inventory::iter` order is unspecified, so everything is sorted
//! (dependency-topological for types, alphabetical to break ties). A generator whose output
//! churns between runs makes every diff useless.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use mm_engine::ffi::{FfiFnDesc, FfiHandleDesc, FfiType};
use mm_engine::game::mirror::{
    FfiAliasDesc, FfiConstDesc, FfiFieldDesc, FfiKind, FfiMirrorDesc, FfiVariantDesc,
};

// -----------------------------------------------------------------------------------
// the registry, resolved
// -----------------------------------------------------------------------------------

/// Every registry, read once and indexed by name.
struct Registry {
    types: BTreeMap<&'static str, &'static FfiMirrorDesc>,
    aliases: BTreeMap<&'static str, &'static FfiAliasDesc>,
    consts: BTreeMap<&'static str, usize>,
    fns: Vec<&'static FfiFnDesc>,
    handles: Vec<&'static FfiHandleDesc>,
}

/// The ABI primitives a field may bottom out in, and the `ctypes` scalar for each. `u64`
/// has no field today; it is here because `HandshakeRequest`'s response is one and the
/// closure test already admits it.
const PRIMITIVES: &[(&str, &str)] = &[
    ("f32", "ctypes.c_float"),
    ("u32", "ctypes.c_uint32"),
    ("i32", "ctypes.c_int32"),
    ("u8", "ctypes.c_uint8"),
    ("u64", "ctypes.c_uint64"),
    ("i64", "ctypes.c_int64"),
    ("bool", "ctypes.c_bool"),
];

/// The Python annotation a primitive reads back as. `ctypes` converts scalar fields to
/// plain Python values on access, so the annotation is `float`/`int`/`bool`, not the
/// `ctypes` class -- annotating them as the latter is the mistake the hand-written stubs in
/// `core/*.pyi` make today.
fn primitive_annotation(ty: &str) -> &'static str {
    match ty {
        "f32" => "float",
        "bool" => "bool",
        _ => "int",
    }
}

fn ctypes_primitive(ty: &str) -> Option<&'static str> {
    PRIMITIVES.iter().find(|(n, _)| *n == ty).map(|(_, c)| *c)
}

impl Registry {
    fn read() -> Self {
        let mut types = BTreeMap::new();
        for d in inventory::iter::<FfiMirrorDesc> {
            assert!(types.insert(d.name, d).is_none(), "{} is registered twice", d.name);
        }
        let mut aliases = BTreeMap::new();
        for a in inventory::iter::<FfiAliasDesc> {
            assert!(aliases.insert(a.name, a).is_none(), "{} is aliased twice", a.name);
        }
        let mut consts = BTreeMap::new();
        for c in inventory::iter::<FfiConstDesc> {
            assert!(consts.insert(c.name, c.value).is_none(), "{} is declared twice", c.name);
        }
        let mut fns: Vec<_> = inventory::iter::<FfiFnDesc>.into_iter().collect();
        fns.sort_by_key(|f| f.name);
        let mut handles: Vec<_> = inventory::iter::<FfiHandleDesc>.into_iter().collect();
        handles.sort_by_key(|h| h.name);

        // A registry this small can only be short for one reason, and it is not a small
        // one: `inventory` registers through link-section statics, and a linker that drops
        // an unreferenced object file drops the registrations in it without a word. Better
        // to stop here than to write a plausible-looking half of the bindings.
        assert!(
            types.len() >= 25 && !fns.is_empty() && !handles.is_empty() && consts.len() == 11,
            "the registry came back short ({} types, {} fns, {} handles, {} consts) -- \
             the linker has dropped registrations; see native/src/lib.rs",
            types.len(),
            fns.len(),
            handles.len(),
            consts.len(),
        );

        Registry { types, aliases, consts, fns, handles }
    }

    /// Substitutes a generic parameter for the argument the instantiation was registered
    /// with. An instantiated descriptor (`StateOption<Vec2>`) carries layout measured for
    /// that monomorphization but fields still spelled `T`.
    fn substitute<'a>(&self, owner: &FfiMirrorDesc, ty: &'a str) -> &'a str
    where
        'static: 'a,
    {
        match owner.generic_params.iter().position(|p| *p == ty) {
            Some(i) => owner.generic_args[i],
            None => ty,
        }
    }
}

// -----------------------------------------------------------------------------------
// type spellings
// -----------------------------------------------------------------------------------

/// A field's type, resolved down to a shape the emitter can write.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Resolved {
    /// The element spelling: a primitive, or a registered type name.
    element: String,
    /// Array dimensions, **outermost first** -- `[[MapTile; MAP_SIZE]; MAP_SIZE]` gives
    /// `[("MAP_SIZE", 32), ("MAP_SIZE", 32)]`. Empty for a scalar field.
    ///
    /// Both the spelling and the value: the emitter writes the spelling, so generated
    /// Python reads `MapTile_t * MAP_SIZE * MAP_SIZE` rather than a bare `32` nobody can
    /// trace, while `layout.json` and the size cross-check use the value.
    dims: Vec<(String, usize)>,
}

/// Splits an array spelling into its element and its dimension *names*, outermost first.
///
/// The derive normalizes the spelling (`mm-macros/src/ffi_mirror.rs::spell`: whitespace
/// stripped, then one space after every `;` and `,`), so this is an exact parse rather
/// than a heuristic. `[[MapTile; MAP_SIZE]; MAP_SIZE]` peels outside-in: the last `;` in
/// the body separates the outermost length from everything it contains.
fn peel_array(mut ty: &str) -> (&str, Vec<&str>) {
    let mut dims = Vec::new();
    while let Some(inner) = ty.strip_prefix('[') {
        let inner = inner.strip_suffix(']').unwrap_or(inner);
        let Some((elem, len)) = inner.rsplit_once(';') else {
            break;
        };
        dims.push(len.trim());
        ty = elem.trim();
    }
    (ty, dims)
}

impl Registry {
    /// Resolves a field's written spelling: substitutes generics, follows aliases, and
    /// peels arrays into concrete dimensions.
    fn resolve(&self, owner: &FfiMirrorDesc, field: &FfiFieldDesc) -> Resolved {
        let spelling = self.substitute(owner, field.ty);
        let mut dims: Vec<(String, usize)> = Vec::new();
        let mut ty = spelling;

        // An alias may itself be an array (`Map`), so alternate peeling and following
        // until neither applies. Two passes is all the closure needs, but the loop costs
        // nothing and does not care.
        loop {
            let (element, names) = peel_array(ty);
            for n in names {
                let value = *self.consts.get(n).unwrap_or_else(|| {
                    panic!(
                        "{}::{} is a `{}`, whose length `{n}` is not registered -- \
                         add it to `mm_ffi_const!` in engine/src/ffi.rs",
                        owner.name, field.name, field.ty,
                    )
                });
                dims.push((n.to_string(), value));
            }
            match self.aliases.get(element) {
                // A registered type wins over an alias of the same name; an alias to a
                // primitive (`BotId = u8`) resolves through to the primitive, because
                // Python has no distinct type to give it.
                Some(alias) if !self.types.contains_key(element) => {
                    if alias.target == element {
                        break;
                    }
                    ty = alias.target;
                }
                _ => {
                    return Resolved { element: element.to_string(), dims };
                }
            }
        }
        Resolved { element: ty.to_string(), dims }
    }

    /// Every type a descriptor's fields depend on, for the topological sort.
    fn dependencies(&self, desc: &FfiMirrorDesc) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for f in fields_of(desc) {
            let r = self.resolve(desc, f);
            if self.types.contains_key(r.element.as_str()) {
                out.insert(r.element);
            }
        }
        out
    }
}

/// Every field of a type, variants flattened. A data enum's variant fields carry offsets
/// that are already absolute within the enum.
fn fields_of(desc: &FfiMirrorDesc) -> Vec<&'static FfiFieldDesc> {
    match desc.kind {
        FfiKind::Struct { fields } => fields.iter().collect(),
        FfiKind::UnitEnum { .. } => vec![],
        FfiKind::DataEnum { variants, .. } => {
            variants.iter().flat_map(|v| v.fields.iter()).collect()
        }
    }
}

// -----------------------------------------------------------------------------------
// naming
// -----------------------------------------------------------------------------------

/// `StateOption<Vec2>` -> `StateOption_Vec2`. Python has no generics at the `ctypes` level,
/// so an instantiation becomes its own class and the mangled name is how a field spelled
/// `StateOption<Vec2>` finds it.
fn class_name(ty: &str) -> String {
    ty.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect::<String>()
        .trim_matches('_')
        .replace("__", "_")
}

/// The `ctypes` scalar standing in for a unit enum in a `_fields_` list. `ctypes` rejects
/// an `IntEnum` there, so the enum carries the values and this carries the layout -- the
/// `_t` suffix is the starterpack's existing convention (`Team_t` in `core/state.py`).
fn enum_scalar(ty: &str) -> String {
    format!("{}_t", class_name(ty))
}

/// `TargetPosition` -> `target_position`, for the payload union's member names. Matches the
/// snake-casing `mm-macros` does for the shadow union it measures.
fn snake_case(name: &str) -> String {
    let mut out = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() {
            if i != 0 {
                out.push('_');
            }
            out.extend(ch.to_lowercase());
        } else {
            out.push(ch);
        }
    }
    py_ident(&out)
}

/// Python's reserved words. Rust and Python disagree about what a name may be, and the
/// disagreement is not hypothetical: `StateOption::None` is a real variant, and
/// `None = 0` inside a class body is a `SyntaxError`, not a warning -- the whole generated
/// module would fail to import.
const KEYWORDS: &[&str] = &[
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
    "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
    "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
    "try", "while", "with", "yield", "match", "case",
];

/// A Rust name as a legal Python identifier. A keyword gains a trailing underscore, which
/// is the convention the standard library itself uses (`class_`, `lambda_`). `layout.json`
/// records both spellings, so the parity suite checks the Python name against the Rust one
/// rather than assuming they match.
fn py_ident(name: &str) -> String {
    if KEYWORDS.contains(&name) {
        format!("{name}_")
    } else {
        name.to_string()
    }
}

/// A tuple field is recorded by index (`StateOption::Some(T)` gives a field named `"0"`).
/// Python needs an identifier, so it becomes `f0` -- the same spelling the derive's shadow
/// structs use.
fn field_name(name: &str) -> String {
    if name.chars().all(|c| c.is_ascii_digit()) {
        format!("f{name}")
    } else {
        py_ident(name)
    }
}

/// The payload struct for one variant: `_TurnAction_TargetPosition`.
fn variant_struct(owner: &str, variant: &str) -> String {
    format!("_{}_{}", class_name(owner), variant)
}

// -----------------------------------------------------------------------------------
// the friendly surface
// -----------------------------------------------------------------------------------
//
// Everything above is layout. Everything here is the API `dev/ffi.md` promises -- and it is
// generated rather than hand-written in `core/channel.py` for the reason the whole FFI
// effort exists: a hand-maintained surface over a struct that gains a field drifts, and a
// drifting Python bot does not fail, it reads the wrong bytes.
//
// Two rules do most of the work, and both are derived from the registry:
//
// - a field whose type is a `StateOption<T>` instantiation is emitted under a leading
//   underscore, and the public name becomes an `Optional[T]` property. `_fields_` names are
//   what define offsets, so a friendlier reader has to displace the raw one.
// - every data enum gains a `variant` property, a constructor per variant spelled the way
//   Rust spells it (`TurnAction.TargetPosition(pos=...)`), and an `as_<variant>` accessor.
//
// The rest is semantics a registry cannot know -- that `BotArray` is a collection, that
// `Vec2` is a vector -- so it is declared below, in Python, as a verbatim block per type.
// Emitting fixed Python from a Rust literal is the same thing `SURFACE_HEADER` already does.

/// The `_fields_` entries whose public spelling is taken over by the friendly surface.
///
/// `BotArray` is the only entry: `len`/`mask`/`arr` are two parallel arrays and a count,
/// which is not a thing to hand a competitor -- the class is a sequence over the live slots
/// instead, and the raw trio stays reachable for the parity suite.
const PRIVATE_FIELDS: &[(&str, &[&str])] = &[("BotArray", &["len", "mask", "arr"])];

fn is_private_field(owner: &str, field: &str) -> bool {
    PRIVATE_FIELDS
        .iter()
        .any(|(ty, fields)| *ty == owner && fields.contains(&field))
}

/// Python bodies for the types whose surface is more than their layout, keyed by Rust name.
///
/// Declared, not derived. `BotArray` being a sequence, `Vec2` being a vector and
/// `GameState::fleets()` pairing two fields by team are facts about the game, and no amount
/// of layout metadata implies them. Each block is appended verbatim into the class body
/// after `_fields_`, so it may use the private field names above.
///
/// Every method here mirrors one that already exists on the Rust side, by name and by
/// behaviour -- the point is that a strategy reads the same in both languages. The Rust
/// original is cited in each docstring so the pair can be checked.
const SEMANTIC: &[(&str, &str)] = &[
    ("Vec2", VEC2_BODY),
    ("Team", TEAM_BODY),
    ("BotArray", BOT_ARRAY_BODY),
    ("BotState", BOT_STATE_BODY),
    ("FleetAction", FLEET_ACTION_BODY),
    ("GameState", GAME_STATE_BODY),
    ("TeamPair<u32>", TEAM_PAIR_BODY),
];

fn semantic_body(rust_name: &str) -> Option<&'static str> {
    SEMANTIC.iter().find(|(n, _)| *n == rust_name).map(|(_, b)| *b)
}

/// `engine/src/game/util.rs`'s `Vec2`, method for method. Replaces the starterpack's old
/// hand-written `core/util.py`, whose `normalize()`/`theta()` had no Rust counterpart.
const VEC2_BODY: &str = r#"
    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        """Defaulted, so `Vec2()` is the zero vector -- Rust's `Vec2::ZERO`."""
        super().__init__(x, y)

    @classmethod
    def from_angle_rad(cls, angle_rad: float) -> Vec2:
        return cls(math.cos(angle_rad), math.sin(angle_rad))

    @classmethod
    def from_angle_deg(cls, angle_deg: float) -> Vec2:
        return cls.from_angle_rad(math.radians(angle_deg))

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def norm_sq(self) -> float:
        return self.x * self.x + self.y * self.y

    def norm(self) -> float:
        return math.sqrt(self.norm_sq())

    def angle_rad(self) -> float:
        return math.atan2(self.y, self.x)

    def angle_deg(self) -> float:
        return math.degrees(self.angle_rad())

    def normalize_or_zero(self) -> Vec2:
        """A unit vector in the same direction, or `(0, 0)` if there is no direction."""
        n = self.norm()
        return Vec2(0.0, 0.0) if n == 0.0 else Vec2(self.x / n, self.y / n)

    def rotate_rad(self, angle_rad: float) -> Vec2:
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        return Vec2(self.x * c - self.y * s, self.x * s + self.y * c)

    def rotate_deg(self, angle_deg: float) -> Vec2:
        return self.rotate_rad(math.radians(angle_deg))

    def dist_sq(self, other: Vec2) -> float:
        return (self - other).norm_sq()

    def dist(self, other: Vec2) -> float:
        return (self - other).norm()

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vec2:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vec2:
        return Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self) -> Vec2:
        return Vec2(-self.x, -self.y)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vec2) and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"Vec2({self.x}, {self.y})"
"#;

/// `engine/src/game/team.rs::Team::other_team`.
const TEAM_BODY: &str = r#"
    def other_team(self) -> Team:
        return Team.Other if self is Team.Me else Team.Me
"#;

/// `engine/src/game/state.rs::BotArray`'s `len`/`iter`/`get`/`is_full`, as the sequence
/// protocol. `add`/`remove` are the engine's and are deliberately absent.
const BOT_ARRAY_BODY: &str = r#"
    def __len__(self) -> int:
        """How many slots are live -- not `BOTS_MAX`."""
        return self._len

    def __iter__(self) -> Iterator[BotState]:
        """The live bots, in id order. Dead slots are skipped, not yielded as `None`."""
        for i in range(BOTS_MAX):
            if self._mask[i]:
                yield self._arr[i]

    def __getitem__(self, bot_id: int) -> Optional[BotState]:
        """The bot in slot `bot_id`, or `None` if that slot is dead or out of range.

        Rust's `Index` panics here and `get` returns an `Option`; Python has one syntax, so
        it is the forgiving one. `get` is kept as an alias for symmetry with the Rust name.
        """
        if not 0 <= bot_id < BOTS_MAX or not self._mask[bot_id]:
            return None
        return self._arr[bot_id]

    def get(self, bot_id: int) -> Optional[BotState]:
        return self[bot_id]

    def ids(self) -> Iterator[int]:
        """The live slot numbers, which are also the indices into `FleetAction.bots`."""
        return (i for i in range(BOTS_MAX) if self._mask[i])

    def is_full(self) -> bool:
        return self._len >= BOTS_MAX
"#;

/// `engine/src/game/state.rs::BotState`'s accessors. Each one reads through `special`, and
/// each reports the same nothing-to-say value Rust does for a class that cannot do the
/// thing -- `0` for a blaster that does not exist, `None` for the rest.
const BOT_STATE_BODY: &str = r#"
    @property
    def class_(self) -> BotClass:
        """This bot's class. Trailing underscore because `class` is a Python keyword.

        Resolved by variant *name* rather than by tag number: `SpecialStateTag` and
        `BotClass` happen to agree today, and a rename or reorder on one side should be a
        `KeyError` here rather than a silently wrong class.
        """
        return BotClass[SpecialStateTag(self.special.tag).name]

    @property
    def next_fire_tick(self) -> int:
        """The blaster's readiness tick, or `0` -- "ready now" -- for a class without one."""
        battle = self.special.as_battle
        return 0 if battle is None else battle.next_fire_tick

    @property
    def shot(self) -> Optional[Vec2]:
        """Impact point of this tick's shot; `None` on a tick this bot did not fire."""
        battle = self.special.as_battle
        return None if battle is None else battle.shot

    @property
    def healing(self) -> Optional[int]:
        """The ally this bot healed this tick, by id."""
        healer = self.special.as_healer
        return None if healer is None else healer.healing

    @property
    def extracting(self) -> Optional[Team]:
        """The deposit this bot extracted from this tick, named by its owning team."""
        extractor = self.special.as_extractor
        return None if extractor is None else extractor.extracting
"#;

/// `engine/src/game/state.rs::FleetAction::new`.
const FLEET_ACTION_BODY: &str = r#"
    @classmethod
    def new(cls) -> FleetAction:
        """The do-nothing action: no movement, no specials, no rush order.

        Every Rust default in this struct is its zero value, and `ctypes` zeroes a fresh
        instance, so this is `FleetAction()` -- kept as the name bots construct through, the
        way `FleetAction::new()` is on the Rust side.
        """
        return cls()
"#;

/// `engine/src/game/state.rs`'s `GameState` accessors. The three pair views exist because
/// Rust returns a `TeamPair<&T>` from them, and indexing by `Team` is how a strategy asks
/// about the other side without branching on which side it is.
const GAME_STATE_BODY: &str = r#"
    def fleets(self) -> _Pair[BotArray]:
        return _Pair(self, "fleet_me", "fleet_other")

    def deposits(self) -> _Pair[Deposit]:
        return _Pair(self, "deposit_me", "deposit_other")

    def fabricators(self) -> _Pair[FabricatorState]:
        return _Pair(self, "fabricator_me", "fabricator_other")

    def payload_pos(self) -> Vec2:
        """Centre of the payload circle this tick. Its radius is `conf.payload.radius`."""
        out = (ctypes.c_float * 2)()
        mm_payload_pos(self.capture, out)
        return Vec2(out[0], out[1])

"#;

/// `engine/src/game/team.rs::TeamPair`'s `Index<Team>`. The `me`/`other` fields stay public
/// -- they are the fast path and the names the Rust client build uses.
const TEAM_PAIR_BODY: &str = r#"
    def __getitem__(self, team: Team) -> int:
        return self.me if team == Team.Me else self.other

    def __setitem__(self, team: Team, value: int) -> None:
        if team == Team.Me:
            self.me = value
        else:
            self.other = value

    def __iter__(self) -> Iterator[int]:
        yield self.me
        yield self.other
"#;

/// Emitted once, before the wire types. `_Pair` is referenced only from method bodies, so
/// it does not need `Team` or any wire class to exist yet.
const PRELUDE: &str = r#"
# -----------------------------------------------------------------------------------
# helpers the friendly surface is built from
# -----------------------------------------------------------------------------------

_T = TypeVar("_T")


class _Pair(Generic[_T]):
    """Two attributes of one struct, indexed by `Team`.

    `GameState` stores its fleets, deposits and fabricators as `*_me`/`*_other` field pairs
    rather than as a `TeamPair`, because a `TeamPair<BotArray>` would be one struct the
    engine's `Diff` could not describe per fleet. This is the `TeamPair<&T>` that Rust's
    `GameState::fleets()` hands back: a view, not a copy, so writing through it is writing
    through to the state.
    """

    __slots__ = ("_owner", "_me_attr", "_other_attr")

    def __init__(self, owner: object, me_attr: str, other_attr: str) -> None:
        self._owner = owner
        self._me_attr = me_attr
        self._other_attr = other_attr

    def _attr(self, team: Team) -> str:
        return self._me_attr if team == Team.Me else self._other_attr

    def __getitem__(self, team: Team) -> _T:
        return getattr(self._owner, self._attr(team))

    def __setitem__(self, team: Team, value: _T) -> None:
        setattr(self._owner, self._attr(team), value)

    @property
    def me(self) -> _T:
        return self[Team.Me]

    @property
    def other(self) -> _T:
        return self[Team.Other]

    def __iter__(self) -> Iterator[_T]:
        yield self.me
        yield self.other

    def __repr__(self) -> str:
        return f"_Pair(me={self.me!r}, other={self.other!r})"
"#;

// -----------------------------------------------------------------------------------
// StateOption, recognized
// -----------------------------------------------------------------------------------

impl Registry {
    /// The `T` of a `StateOption<T>` instantiation, or `None` for anything else.
    ///
    /// Recognized by registered name rather than by shape: a two-variant `#[repr(u8, C)]`
    /// enum with a `None` and a `Some(T)` is not a pattern worth inferring, and
    /// `mm_ffi_instantiate!` spells the name out anyway.
    fn state_option_arg(&self, element: &str) -> Option<&'static str> {
        let desc = self.types.get(element)?;
        if !desc.name.starts_with("StateOption<") {
            return None;
        }
        desc.generic_args.first().copied()
    }

    /// A bare type name with aliases followed through, the way `resolve` does for a field.
    ///
    /// A generic argument arrives as the literal `mm_ffi_instantiate!` spelled, so
    /// `StateOption<BotId>` carries `"BotId"` -- an alias, and annotating it as one would
    /// give Python `Optional[ctypes.c_uint8]` where the value is a plain `int`.
    fn follow_alias<'b>(&self, mut element: &'b str) -> &'b str
    where
        'static: 'b,
    {
        while !self.types.contains_key(element) {
            match self.aliases.get(element) {
                Some(alias) if alias.target != element => element = alias.target,
                _ => break,
            }
        }
        element
    }

    /// Whether this field is read through an `Optional[T]` property instead of directly.
    ///
    /// Only a scalar field: an *array* of `StateOption` has no natural `Optional` view and
    /// none exists in the closure today, so it would be a silent mis-emission rather than a
    /// feature.
    fn optional_field(&self, owner: &FfiMirrorDesc, field: &FfiFieldDesc) -> Option<&'static str> {
        let r = self.resolve(owner, field);
        if !r.dims.is_empty() {
            return None;
        }
        self.state_option_arg(&r.element)
    }

    /// The name a field carries in `_fields_`, which is what fixes its offset.
    ///
    /// A field the friendly surface takes over gains a leading underscore. `layout.json`
    /// records this spelling, so the parity suite keeps checking the real entry rather than
    /// the property in front of it.
    fn wire_field_name(&self, owner: &FfiMirrorDesc, field: &FfiFieldDesc) -> String {
        let base = field_name(field.name);
        let taken = is_private_field(owner.name, field.name)
            || self.optional_field(owner, field).is_some();
        if taken {
            format!("_{base}")
        } else {
            base
        }
    }
}

// -----------------------------------------------------------------------------------
// emission
// -----------------------------------------------------------------------------------

struct Emitter<'a> {
    reg: &'a Registry,
    out: String,
}

impl<'a> Emitter<'a> {
    /// The `ctypes` type to write in a `_fields_` entry, arrays included.
    fn ctypes_of(&self, owner: &FfiMirrorDesc, field: &FfiFieldDesc) -> String {
        let r = self.reg.resolve(owner, field);
        let base = self.ctypes_element(&r.element);
        // `T * inner * outer` builds an array of `outer` arrays of `inner`, matching Rust's
        // `[[T; inner]; outer]` and keeping `map[x][y]` indexing. `dims` is outermost first,
        // so it multiplies in reverse.
        let mut ty = base;
        for (spelling, _) in r.dims.iter().rev() {
            ty = format!("{ty} * {spelling}");
        }
        ty
    }

    fn ctypes_element(&self, element: &str) -> String {
        if let Some(prim) = ctypes_primitive(element) {
            return prim.to_string();
        }
        match self.reg.types.get(element).map(|d| d.kind) {
            // A unit enum is a bare `u8` on the wire; the `IntEnum` beside it carries the
            // names. `ctypes` will not accept the `IntEnum` in `_fields_`.
            Some(FfiKind::UnitEnum { .. }) => enum_scalar(element),
            Some(_) => class_name(element),
            None => panic!("`{element}` resolves to nothing the generator can emit"),
        }
    }

    /// The Python annotation for a field, as `ctypes` hands it back on attribute access.
    fn annotation_of(&self, owner: &FfiMirrorDesc, field: &FfiFieldDesc) -> String {
        let r = self.reg.resolve(owner, field);
        let base = if let Some(_) = ctypes_primitive(&r.element) {
            primitive_annotation(&r.element).to_string()
        } else {
            match self.reg.types.get(r.element.as_str()).map(|d| d.kind) {
                // Reading a `u8`-backed field gives an `int`; it compares equal to the
                // `IntEnum` member, which is the point of `IntEnum`.
                Some(FfiKind::UnitEnum { .. }) => "int".to_string(),
                _ => class_name(&r.element),
            }
        };
        // `ctypes.Array` is generic over the element at type-check time only.
        (0..r.dims.len()).fold(base, |acc, _| format!("ctypes.Array[{acc}]"))
    }

    fn fields_block(&mut self, owner: &FfiMirrorDesc, fields: &[FfiFieldDesc], indent: &str) {
        for f in fields {
            let _ = writeln!(
                self.out,
                "{indent}{}: {}",
                self.reg.wire_field_name(owner, f),
                self.annotation_of(owner, f)
            );
        }
        if fields.is_empty() {
            let _ = writeln!(self.out, "{indent}_fields_ = []");
            return;
        }
        let _ = writeln!(self.out, "{indent}_fields_ = [");
        for f in fields {
            let _ = writeln!(
                self.out,
                "{indent}    ({:?}, {}),",
                self.reg.wire_field_name(owner, f),
                self.ctypes_of(owner, f)
            );
        }
        let _ = writeln!(self.out, "{indent}]");
    }

    fn emit_struct(&mut self, desc: &FfiMirrorDesc, fields: &[FfiFieldDesc]) {
        let name = class_name(desc.name);
        let _ = writeln!(self.out, "\nclass {name}(ctypes.Structure):");
        let _ = writeln!(
            self.out,
            "    \"\"\"`{}` -- {} bytes, align {}.\"\"\"",
            desc.name, desc.size, desc.align
        );
        self.fields_block(desc, fields, "    ");
        self.optional_properties(desc, fields);
        self.semantic(desc.name);
    }

    fn emit_unit_enum(&mut self, desc: &FfiMirrorDesc, variants: &[FfiVariantDesc]) {
        let name = class_name(desc.name);
        let _ = writeln!(self.out, "\nclass {name}(enum.IntEnum):");
        let _ = writeln!(
            self.out,
            "    \"\"\"`{}` -- a `u8` on the wire. `IntEnum`, so a field read back as an\n\
             \x20   `int` compares equal to a member without conversion.\"\"\"",
            desc.name
        );
        for v in variants {
            let _ = writeln!(self.out, "    {} = {}", py_ident(v.name), v.tag);
        }
        self.semantic(desc.name);
        // The layout stand-in: `ctypes` rejects an `IntEnum` in a `_fields_` list.
        let _ = writeln!(self.out, "\n{} = ctypes.c_uint8", enum_scalar(desc.name));
    }

    fn emit_data_enum(
        &mut self,
        desc: &FfiMirrorDesc,
        payload_offset: usize,
        variants: &[FfiVariantDesc],
    ) {
        let name = class_name(desc.name);

        // One `Structure` per variant. Its field offsets are relative to the payload, so
        // the absolute offsets the registry records are shifted back by `payload_offset`.
        for v in variants {
            let sname = variant_struct(desc.name, v.name);
            let _ = writeln!(self.out, "\nclass {sname}(ctypes.Structure):");
            let _ = writeln!(
                self.out,
                "    \"\"\"`{}::{}` -- the payload only; the tag lives in `{name}`.\"\"\"",
                desc.name, v.name
            );
            self.fields_block(desc, v.fields, "    ");
            // A variant payload is reachable on its own (`bot.special.as_battle`), so its
            // `StateOption` fields need the same `Optional` reader the owner's would get.
            self.optional_properties(desc, v.fields);
        }

        // The union of all of them. It is the union, not any single variant, that carries
        // the max-alignment padding -- which is what fixes where the payload starts.
        let _ = writeln!(self.out, "\nclass _{name}_Payload(ctypes.Union):");
        for v in variants {
            let _ = writeln!(
                self.out,
                "    {}: {}",
                snake_case(v.name),
                variant_struct(desc.name, v.name)
            );
        }
        let _ = writeln!(self.out, "    _fields_ = [");
        for v in variants {
            let _ = writeln!(
                self.out,
                "        ({:?}, {}),",
                snake_case(v.name),
                variant_struct(desc.name, v.name)
            );
        }
        let _ = writeln!(self.out, "    ]");

        let _ = writeln!(self.out, "\nclass {name}Tag(enum.IntEnum):");
        for v in variants {
            let _ = writeln!(self.out, "    {} = {}", py_ident(v.name), v.tag);
        }

        let _ = writeln!(self.out, "\nclass {name}(ctypes.Structure):");
        let _ = writeln!(
            self.out,
            "    \"\"\"`{}` -- `#[repr(u8, C)]`: a `u8` tag at 0, payload at {}.\n\
             \x20   {} bytes, align {}. Read `tag` against `{name}Tag`, then the matching\n\
             \x20   member of `payload`.\"\"\"",
            desc.name, payload_offset, desc.size, desc.align
        );
        let _ = writeln!(self.out, "    tag: int");
        let _ = writeln!(self.out, "    payload: _{name}_Payload");
        let _ = writeln!(self.out, "    _fields_ = [");
        let _ = writeln!(self.out, "        (\"tag\", ctypes.c_uint8),");
        let _ = writeln!(self.out, "        (\"payload\", _{name}_Payload),");
        let _ = writeln!(self.out, "    ]");

        // `StateOption` is the one data enum that gets no tagged surface. `ffi.md` is
        // explicit that it presents as a plain `Optional[T]`, and a `Some`/`None_`
        // constructor pair beside `of()` would be three ways to say one thing.
        match self.reg.state_option_arg(desc.name) {
            Some(arg) => self.state_option_members(desc, arg),
            None => self.data_enum_members(desc, variants),
        }
        self.semantic(desc.name);
    }

    /// The annotation a caller sees for a field, which is the `Optional[T]` a `StateOption`
    /// field presents rather than the wrapper class behind it.
    fn public_annotation_of(&self, owner: &FfiMirrorDesc, field: &FfiFieldDesc) -> String {
        match self.reg.optional_field(owner, field) {
            Some(arg) => format!("Optional[{}]", self.element_annotation(arg)),
            None => self.annotation_of(owner, field),
        }
    }

    /// The annotation for a bare element spelling, as the friendly surface hands it back.
    ///
    /// Differs from `annotation_of` on one point: a unit enum reads back as an `int` through
    /// `ctypes`, but `StateOption<Team>.option` converts it, so the annotation is the enum.
    fn element_annotation(&self, element: &str) -> String {
        let element = self.reg.follow_alias(element);
        if ctypes_primitive(element).is_some() {
            return primitive_annotation(element).to_string();
        }
        class_name(element)
    }

    /// `Optional[T]` properties for every `StateOption` field of a struct or variant.
    fn optional_properties(&mut self, owner: &FfiMirrorDesc, fields: &[FfiFieldDesc]) {
        for f in fields {
            let Some(arg) = self.reg.optional_field(owner, f) else {
                continue;
            };
            let public = field_name(f.name);
            let wire = self.reg.wire_field_name(owner, f);
            let ann = format!("Optional[{}]", self.element_annotation(arg));
            let option_class = class_name(&self.reg.resolve(owner, f).element);
            let _ = write!(
                self.out,
                "
    @property
    def {public}(self) -> {ann}:
        return self.{wire}.option

    @{public}.setter
    def {public}(self, value: {ann}) -> None:
        self.{wire} = {option_class}.of(value)
"
            );
        }
    }

    /// The semantic block declared for this type, if any.
    fn semantic(&mut self, rust_name: &str) {
        if let Some(body) = semantic_body(rust_name) {
            self.out.push_str(body);
        }
    }

    /// `StateOption<T>`'s own two members: the `Optional[T]` read and the constructor.
    ///
    /// Emitted on the wrapper class as well as on every field that holds one, because a
    /// `SpecialState` payload struct is reachable directly (`bot.special.as_battle`) and its
    /// `shot` has to behave the same way there.
    fn state_option_members(&mut self, desc: &FfiMirrorDesc, arg: &str) {
        let name = class_name(desc.name);
        let ann = self.element_annotation(arg);
        // A unit enum reads back from `ctypes` as a plain `int`; wrap it so the caller gets
        // the member. A struct or primitive already arrives as itself.
        let arg = self.reg.follow_alias(arg);
        let convert = match self.reg.types.get(arg).map(|d| d.kind) {
            Some(FfiKind::UnitEnum { .. }) => format!("{}(", class_name(arg)),
            _ => String::new(),
        };
        let close = if convert.is_empty() { "" } else { ")" };
        let _ = write!(
            self.out,
            "
    @property
    def option(self) -> Optional[{ann}]:
        \"\"\"`StateOption<{arg}>` as Python sees it. Rust's `StateOption::option`.\"\"\"
        if self.tag != {name}Tag.Some:
            return None
        return {convert}self.payload.some.f0{close}

    @classmethod
    def of(cls, value: Optional[{ann}]) -> {name}:
        \"\"\"The wrapper around `value`, or the `None` variant. Rust's `From<Option<T>>`.\"\"\"
        out = cls()
        if value is None:
            out.tag = {name}Tag.None_
        else:
            out.tag = {name}Tag.Some
            out.payload.some.f0 = value
        return out

    def __repr__(self) -> str:
        return repr(self.option)
"
        );
    }

    /// The generic data-enum surface: which variant this is, a constructor per variant, and
    /// an accessor per variant that answers `None` when the tag says otherwise.
    fn data_enum_members(&mut self, desc: &FfiMirrorDesc, variants: &[FfiVariantDesc]) {
        let name = class_name(desc.name);
        let _ = write!(
            self.out,
            "
    @property
    def variant(self) -> {name}Tag:
        return {name}Tag(self.tag)

    def __repr__(self) -> str:
        return f\"{name}.{{{name}Tag(self.tag).name}}\"
"
        );

        for v in variants {
            let ctor = py_ident(v.name);
            let member = snake_case(v.name);
            let params: Vec<String> = v
                .fields
                .iter()
                .map(|f| format!("{}: {}", field_name(f.name), self.public_annotation_of(desc, f)))
                .collect();
            let _ = write!(
                self.out,
                "
    @classmethod
    def {ctor}(cls{}{}) -> {name}:
        \"\"\"`{}::{}`, tag and payload set together.\"\"\"
        out = cls()
        out.tag = {name}Tag.{ctor}
",
                if params.is_empty() { "" } else { ", " },
                params.join(", "),
                desc.name,
                v.name,
            );
            if !v.fields.is_empty() {
                let _ = writeln!(self.out, "        payload = out.payload.{member}");
                for f in v.fields {
                    let n = field_name(f.name);
                    let _ = writeln!(self.out, "        payload.{n} = {n}");
                }
            }
            let _ = writeln!(self.out, "        return out");

            let _ = write!(
                self.out,
                "
    @property
    def as_{member}(self) -> Optional[{}]:
        \"\"\"The `{}` payload, or `None` if this value is a different variant.\"\"\"
        if self.tag != {name}Tag.{ctor}:
            return None
        return self.payload.{member}
",
                variant_struct(desc.name, v.name),
                v.name,
            );
        }
    }
}

/// Dependency order, alphabetical within a level. `inventory::iter` order is unspecified,
/// and Python needs a class defined before a `_fields_` entry names it.
fn topological(reg: &Registry) -> Vec<&'static FfiMirrorDesc> {
    let mut emitted: BTreeSet<&str> = BTreeSet::new();
    let mut out = Vec::new();
    let mut pending: Vec<&FfiMirrorDesc> = reg.types.values().copied().collect();

    while !pending.is_empty() {
        let before = pending.len();
        let mut next = Vec::new();
        for desc in pending {
            if reg
                .dependencies(desc)
                .iter()
                .all(|d| d.as_str() == desc.name || emitted.contains(d.as_str()))
            {
                emitted.insert(desc.name);
                out.push(desc);
            } else {
                next.push(desc);
            }
        }
        pending = next;
        assert!(
            pending.len() < before,
            "the wire types have a dependency cycle: {:?}",
            pending.iter().map(|d| d.name).collect::<Vec<_>>()
        );
    }
    out
}

// -----------------------------------------------------------------------------------
// the C surface
// -----------------------------------------------------------------------------------

impl<'a> Emitter<'a> {
    /// The `ctypes` type for one `extern "C"` argument or return.
    fn abi_type(&self, ty: FfiType) -> String {
        match ty {
            FfiType::Void => "None".into(),
            FfiType::F32 => "ctypes.c_float".into(),
            FfiType::U32 => "ctypes.c_uint32".into(),
            FfiType::I32 => "ctypes.c_int32".into(),
            FfiType::U8 => "ctypes.c_uint8".into(),
            FfiType::Bool => "ctypes.c_bool".into(),
            // A handle is opaque by construction: Python holds the pointer and hands it
            // back. It crosses as its own `c_void_p` subclass rather than a bare
            // `c_void_p` -- ABI-identical, but `ctypes` then hands a return value back as
            // an `MmChannel` instead of a raw int, and a caller cannot pass one handle type
            // where another was wanted.
            FfiType::Ptr { pointee, .. } if self.reg.handles.iter().any(|h| h.name == pointee) => {
                pointee.to_string()
            }
            FfiType::Ptr { pointee, .. } => match ctypes_primitive(pointee) {
                Some(prim) => format!("ctypes.POINTER({prim})"),
                None if self.reg.types.contains_key(pointee) => {
                    format!("ctypes.POINTER({})", class_name(pointee))
                }
                None => "ctypes.c_void_p".into(),
            },
        }
    }

    /// The Python annotation a caller sees for an argument or return.
    fn abi_annotation(&self, ty: FfiType) -> String {
        match ty {
            FfiType::Void => "None".into(),
            FfiType::F32 => "float".into(),
            FfiType::Bool => "bool".into(),
            FfiType::U32 | FfiType::I32 | FfiType::U8 => "int".into(),
            // A handle comes back as its own class, which is already a type. A scalar or
            // struct pointer comes back as `ctypes.POINTER(X)`, which is a *call* and not
            // a type -- typeshed spells that `ctypes._Pointer[X]`, legal here only because
            // the generated annotations are lazy.
            FfiType::Ptr { .. } => {
                let runtime = self.abi_type(ty);
                match runtime.strip_prefix("ctypes.POINTER(") {
                    Some(inner) => format!("ctypes._Pointer[{}]", inner.trim_end_matches(')')),
                    None => runtime,
                }
            }
        }
    }

    fn emit_entry_points(&mut self) {
        let _ = writeln!(self.out, "\n{}", SURFACE_HEADER);

        for h in &self.reg.handles {
            let _ = writeln!(
                self.out,
                "\nclass {}(ctypes.c_void_p):\n    \
                 \"\"\"Opaque handle. Reclaim it with `{}`.\"\"\"",
                h.name, h.free
            );
        }

        let _ = writeln!(self.out, "\n_lib = ctypes.CDLL(str(_library_path()))\n");
        for f in &self.reg.fns {
            let args: Vec<String> = f.args.iter().map(|(_, t)| self.abi_type(*t)).collect();
            let _ = writeln!(self.out, "_lib.{}.argtypes = [{}]", f.name, args.join(", "));
            let _ = writeln!(
                self.out,
                "_lib.{}.restype = {}",
                f.name,
                self.abi_type(f.ret)
            );
        }

        // A handle's `_free` is registered as an `FfiHandleDesc`, not an `FfiFnDesc`, so the
        // loop above never sees it -- it would otherwise work only through untyped `CDLL`
        // attribute access. Bound here so it is a real entry point like the rest.
        for h in &self.reg.handles {
            let _ = writeln!(self.out, "_lib.{}.argtypes = [{}]", h.free, h.name);
            let _ = writeln!(self.out, "_lib.{}.restype = None", h.free);
        }

        // Thin wrappers, so callers get names, annotations and editor completion rather
        // than attribute access on a `CDLL`.
        for f in &self.reg.fns {
            let params: Vec<String> = f
                .args
                .iter()
                .map(|(n, t)| format!("{n}: {}", self.abi_annotation(*t)))
                .collect();
            let names: Vec<&str> = f.args.iter().map(|(n, _)| *n).collect();
            let _ = writeln!(
                self.out,
                "\ndef {}({}) -> {}:\n    return _lib.{}({})",
                f.name,
                params.join(", "),
                self.abi_annotation(f.ret),
                f.name,
                names.join(", ")
            );
        }
        for h in &self.reg.handles {
            let _ = writeln!(
                self.out,
                "\ndef {}(handle: {}) -> None:\n    return _lib.{}(handle)",
                h.free, h.name, h.free
            );
        }
    }
}

/// Emitted verbatim: locating the shared library is a runtime concern with no registry
/// input, so it is a literal rather than something the generator composes.
const SURFACE_HEADER: &str = r#"# -----------------------------------------------------------------------------------
# the C surface
# -----------------------------------------------------------------------------------


_LIBRARY_NAMES = (
    "libmm_python_native.so",
    "libmm_python_native.dylib",
    "mm_python_native.dll",
)


def _library_path() -> pathlib.Path:
    """The shared library this package's bindings were generated against.

    Two answers, because a bot runs in two very different shapes.

    `$MM_NATIVE_LIB` wins when set. A submitted bot is a zipapp, and `ctypes.CDLL` cannot
    open a path inside a `.pyz` -- nor is `__file__` a real path in there -- so the API's
    compile step drops the library beside `bot.pyz` and the `/out/bot` wrapper points this
    variable at it. Loading from a read-only bind mount is fine; `dlopen` only needs a path.

    Otherwise, the `mm-cli run` local loop: `native/target/release/`, resolved relative to
    this file rather than by name so a bot run from any working directory finds it.
    """
    override = os.environ.get("MM_NATIVE_LIB")
    if override:
        candidate = pathlib.Path(override)
        if not candidate.exists():
            raise FileNotFoundError(
                f"MM_NATIVE_LIB points at {candidate}, which does not exist"
            )
        return candidate

    root = pathlib.Path(__file__).resolve().parent.parent.parent / "native" / "target" / "release"
    for name in _LIBRARY_NAMES:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"no native library under {root} -- run mm-cli run first"
    )


_handle: object = None


def attach(handle: object) -> None:
    """Register the open channel, so the methods that need one can reach it.

    No generated method needs it today; it is the route a method on a state struct would
    take to reach the config the handshake captured, since it has no other way to the
    channel that delivered it. `core/channel.py` calls this once, at open, which is
    the same shape as `get_config()`: one match, one channel, one module global.
    """
    global _handle
    _handle = handle


def detach() -> None:
    global _handle
    _handle = None


def _require_handle() -> object:
    if _handle is None:
        raise RuntimeError(
            "no channel is attached -- open one through core.channel.EngineChannel "
            "before asking the engine about the match config"
        )
    return _handle
"#;

// -----------------------------------------------------------------------------------
// layout.json
// -----------------------------------------------------------------------------------

/// The registry, dumped for `tools/tests/test_python_bindings.py` to check the emitted
/// `ctypes` classes against.
///
/// Hand-rolled rather than `serde_json`: the shim crate has no serde dependency, adding one
/// to a crate every Python competitor compiles is not worth it for one debug artifact, and
/// the shape here is five flat lists.
fn layout_json(reg: &Registry) -> String {
    fn esc(s: &str) -> String {
        s.replace('\\', "\\\\").replace('"', "\\\"")
    }
    fn field(f: &FfiFieldDesc, owner: &FfiMirrorDesc, reg: &Registry) -> String {
        let r = reg.resolve(owner, f);
        let dims: Vec<String> = r.dims.iter().map(|(_, d)| d.to_string()).collect();
        format!(
            r#"{{"name": "{}", "python_name": "{}", "ty": "{}", "element": "{}", "dims": [{}], "offset": {}, "size": {}, "align": {}}}"#,
            esc(f.name),
            esc(&reg.wire_field_name(owner, f)),
            esc(f.ty),
            esc(&r.element),
            dims.join(", "),
            f.offset,
            f.size,
            f.align
        )
    }

    let mut types = Vec::new();
    for desc in topological(reg) {
        let common = format!(
            r#""name": "{}", "python_name": "{}", "size": {}, "align": {}"#,
            esc(desc.name),
            esc(&class_name(desc.name)),
            desc.size,
            desc.align
        );
        let body = match desc.kind {
            FfiKind::Struct { fields } => {
                let fs: Vec<String> = fields.iter().map(|f| field(f, desc, reg)).collect();
                format!(r#""kind": "struct", "fields": [{}]"#, fs.join(", "))
            }
            FfiKind::UnitEnum { variants } => {
                let vs: Vec<String> = variants
                    .iter()
                    .map(|v| format!(r#"{{"name": "{}", "tag": {}}}"#, esc(v.name), v.tag))
                    .collect();
                format!(r#""kind": "unit_enum", "variants": [{}]"#, vs.join(", "))
            }
            FfiKind::DataEnum { payload_offset, variants } => {
                let vs: Vec<String> = variants
                    .iter()
                    .map(|v| {
                        let fs: Vec<String> =
                            v.fields.iter().map(|f| field(f, desc, reg)).collect();
                        format!(
                            r#"{{"name": "{}", "python_name": "{}", "struct": "{}", "tag": {}, "fields": [{}]}}"#,
                            esc(v.name),
                            esc(&snake_case(v.name)),
                            esc(&variant_struct(desc.name, v.name)),
                            v.tag,
                            fs.join(", ")
                        )
                    })
                    .collect();
                format!(
                    r#""kind": "data_enum", "payload_offset": {}, "variants": [{}]"#,
                    payload_offset,
                    vs.join(", ")
                )
            }
        };
        types.push(format!("{{{common}, {body}}}"));
    }

    let aliases: Vec<String> = reg
        .aliases
        .values()
        .map(|a| {
            format!(
                r#"{{"name": "{}", "target": "{}", "size": {}, "align": {}}}"#,
                esc(a.name),
                esc(a.target),
                a.size,
                a.align
            )
        })
        .collect();
    let consts: Vec<String> = reg
        .consts
        .iter()
        .map(|(n, v)| format!(r#"{{"name": "{}", "value": {}}}"#, esc(n), v))
        .collect();
    let fns: Vec<String> = reg
        .fns
        .iter()
        .map(|f| format!(r#"{{"name": "{}"}}"#, esc(f.name)))
        .collect();
    let handles: Vec<String> = reg
        .handles
        .iter()
        .map(|h| {
            format!(
                r#"{{"name": "{}", "free": "{}"}}"#,
                esc(h.name),
                esc(h.free)
            )
        })
        .collect();

    format!(
        "{{\n  \"types\": [\n    {}\n  ],\n  \"aliases\": [{}],\n  \"consts\": [{}],\n  \"functions\": [{}],\n  \"handles\": [{}]\n}}\n",
        types.join(",\n    "),
        aliases.join(", "),
        consts.join(", "),
        fns.join(", "),
        handles.join(", ")
    )
}

// -----------------------------------------------------------------------------------
// driver
// -----------------------------------------------------------------------------------

const BANNER: &str = "\
# Generated by mm-ffi-codegen from the engine's layout registry. DO NOT EDIT.
#
# Every class here mirrors a Rust type's exact `#[repr(C)]` layout -- sizes, alignments and
# field offsets are the ones the engine measured, and `tools/tests/` checks that `ctypes`
# reproduces them. Regenerate with `mm-cli run` after any engine change; this file is
# gitignored precisely so it cannot go stale.
#
# The wire layer and the friendly surface on top of it are both here: `StateOption` fields
# read as `Optional[T]`, `BotArray` is a sequence over its live slots, `TeamPair` and
# `GameState.fleets()` index by `Team`, and a data enum has a constructor per variant. The
# raw `_fields_` entry behind a property keeps its name with a leading underscore.
#
# `core/channel.py` holds what a registry cannot describe: the channel loop and the
# navigation wrappers.
";

fn emit_bindings(reg: &Registry) -> String {
    let mut em = Emitter { reg, out: String::new() };
    em.out.push_str(BANNER);
    // `from __future__ import annotations` makes every annotation below a lazy string.
    // That matters twice: `ctypes.Array[int]` and `ctypes._Pointer[GameConfig]` are valid
    // to a type checker but not expressions Python evaluates, and a class-body annotation
    // on a `ctypes.Structure` would otherwise run at import time.
    em.out.push_str(
        "\nfrom __future__ import annotations\n\n\
         import ctypes\n\
         import enum\n\
         import math\n\
         import os\n\
         import pathlib\n\
         from typing import Generic, Iterator, Optional, TypeVar\n",
    );
    em.out.push_str(PRELUDE);

    em.out.push_str(
        "\n# -----------------------------------------------------------------------------------\n\
         # constants\n\
         # -----------------------------------------------------------------------------------\n",
    );
    for (name, value) in &reg.consts {
        let _ = writeln!(em.out, "{name} = {value}");
    }

    em.out.push_str(
        "\n# -----------------------------------------------------------------------------------\n\
         # wire types, in dependency order\n\
         # -----------------------------------------------------------------------------------\n",
    );
    for desc in topological(reg) {
        match desc.kind {
            FfiKind::Struct { fields } => em.emit_struct(desc, fields),
            FfiKind::UnitEnum { variants } => em.emit_unit_enum(desc, variants),
            FfiKind::DataEnum { payload_offset, variants } => {
                em.emit_data_enum(desc, payload_offset, variants)
            }
        }
    }

    // Aliases last: `Map` names `MapTile_t`, and an alias is only ever referred to by a
    // field through its resolved spelling, so nothing above needs it.
    em.out.push_str(
        "\n# -----------------------------------------------------------------------------------\n\
         # aliases\n\
         # -----------------------------------------------------------------------------------\n",
    );
    for alias in reg.aliases.values() {
        let (element, dim_names) = peel_array(alias.target);
        let base = em.ctypes_element(element);
        let mut ty = base;
        for n in dim_names.iter().rev() {
            ty = format!("{ty} * {n}");
        }
        let _ = writeln!(em.out, "{} = {ty}", class_name(alias.name));
    }

    em.emit_entry_points();
    em.out
}

fn main() {
    let reg = Registry::read();

    let out_dir = parse_out_dir();
    std::fs::create_dir_all(&out_dir).expect("cannot create the output directory");

    write(&out_dir.join("bindings.py"), &emit_bindings(&reg));
    write(
        &out_dir.join("__init__.py"),
        &format!("{BANNER}\nfrom .bindings import *  # noqa: F401,F403\n"),
    );
    write(&out_dir.join("layout.json"), &layout_json(&reg));

    println!(
        "mm-ffi-codegen: {} types, {} aliases, {} constants, {} entry points -> {}",
        reg.types.len(),
        reg.aliases.len(),
        reg.consts.len(),
        reg.fns.len(),
        out_dir.display()
    );
}

fn parse_out_dir() -> PathBuf {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => {
                return PathBuf::from(args.next().expect("--out needs a directory"));
            }
            other => panic!("unknown argument `{other}`; usage: mm-ffi-codegen [--out <dir>]"),
        }
    }
    // Beside the starterpack's `core/`, which is where `core/channel.py` imports from.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../core/_generated")
}

fn write(path: &Path, contents: &str) {
    std::fs::write(path, contents)
        .unwrap_or_else(|e| panic!("cannot write {}: {e}", path.display()));
}

// -----------------------------------------------------------------------------------

#[cfg(test)]
mod codegen_test {
    use super::*;

    /// The spelling parser, on every shape the derive actually produces. A bug here writes
    /// plausible-looking Python that reads the wrong bytes, which is the failure mode this
    /// whole file exists to prevent, so it is worth testing without a Python interpreter in
    /// the loop.
    #[test]
    fn an_array_spelling_peels_outermost_first() {
        assert_eq!(peel_array("f32"), ("f32", vec![]));
        assert_eq!(peel_array("Vec2"), ("Vec2", vec![]));
        assert_eq!(peel_array("StateOption<Vec2>"), ("StateOption<Vec2>", vec![]));
        assert_eq!(peel_array("[BotAction; BOTS_MAX]"), ("BotAction", vec!["BOTS_MAX"]));
        assert_eq!(peel_array("[Vec2; PAYLOAD_PATH_LEN]"), ("Vec2", vec!["PAYLOAD_PATH_LEN"]));
        assert_eq!(
            peel_array("[[MapTile; MAP_SIZE]; MAP_SIZE]"),
            ("MapTile", vec!["MAP_SIZE", "MAP_SIZE"])
        );
    }

    /// The dimension order is the one thing here with no test coverage from the shipped
    /// map, because `Map` is square. `T * inner * outer` in `ctypes` is an array of `outer`
    /// arrays of `inner` -- the same shape as Rust's `[[T; inner]; outer]` -- so the
    /// emitter multiplies `dims` in reverse. Get it backwards and `map[x][y]` silently
    /// transposes the day the map stops being square.
    #[test]
    fn a_nested_array_keeps_rusts_index_order() {
        let (element, dims) = peel_array("[[MapTile; 4]; 7]");
        assert_eq!(element, "MapTile");
        assert_eq!(dims, vec!["7", "4"], "outermost dimension first");

        // What the emitter writes, reversed back out of that.
        let written: Vec<&str> = dims.iter().rev().copied().collect();
        assert_eq!(written, vec!["4", "7"], "`MapTile_t * 4 * 7`: inner dimension first");
    }

    #[test]
    fn a_name_python_cannot_spell_is_mangled() {
        // `StateOption::None` is real, and `None = 0` in a class body is a SyntaxError.
        assert_eq!(py_ident("None"), "None_");
        assert_eq!(py_ident("class"), "class_");
        assert_eq!(py_ident("Some"), "Some");

        // Tuple fields are recorded by index.
        assert_eq!(field_name("0"), "f0");
        assert_eq!(field_name("pos"), "pos");

        assert_eq!(class_name("StateOption<Vec2>"), "StateOption_Vec2");
        assert_eq!(class_name("TeamPair<u32>"), "TeamPair_u32");
        assert_eq!(class_name("GameState"), "GameState");
        assert_eq!(enum_scalar("Team"), "Team_t");

        assert_eq!(snake_case("TargetPosition"), "target_position");
        assert_eq!(snake_case("Battle"), "battle");
        // A variant that snake-cases into a keyword still has to be spellable.
        assert_eq!(snake_case("None"), "none");
    }

    /// Resolution against the real registry, which is the only place generic substitution
    /// and alias-following can be exercised honestly.
    #[test]
    fn a_field_resolves_through_generics_and_aliases() {
        let reg = Registry::read();
        let of = |ty: &str, field: &str| {
            let desc = reg.types[ty];
            let f = fields_of(desc)
                .into_iter()
                .find(|f| f.name == field)
                .unwrap_or_else(|| panic!("{ty} has no field {field}"));
            reg.resolve(desc, f)
        };

        // A plain struct field.
        assert_eq!(of("Vec2", "x"), Resolved { element: "f32".into(), dims: vec![] });

        // `BotId` is an alias of `u8`; Python has no distinct type for it, so it resolves
        // through to the primitive.
        assert_eq!(of("BotState", "id"), Resolved { element: "u8".into(), dims: vec![] });

        // `Map` is an alias whose target is itself a nested array.
        assert_eq!(
            of("GameConfig", "map"),
            Resolved {
                element: "MapTile".into(),
                dims: vec![("MAP_SIZE".into(), 32), ("MAP_SIZE".into(), 32)],
            }
        );

        // A one-dimensional array of a registered struct.
        assert_eq!(
            of("FleetAction", "bots"),
            Resolved { element: "BotAction".into(), dims: vec![("BOTS_MAX".into(), 32)] }
        );

        // The substitution that instantiations exist for: the descriptor's field is still
        // spelled `T`, and only `generic_args` says which `T`.
        assert_eq!(
            of("StateOption<Vec2>", "0"),
            Resolved { element: "Vec2".into(), dims: vec![] }
        );
        assert_eq!(
            of("StateOption<BotId>", "0"),
            Resolved { element: "u8".into(), dims: vec![] }
        );
    }

    /// Python needs a class defined before a `_fields_` entry names it, and
    /// `inventory::iter` order is unspecified.
    #[test]
    fn types_are_emitted_after_everything_they_name() {
        let reg = Registry::read();
        let order = topological(&reg);
        assert_eq!(order.len(), reg.types.len(), "every type is emitted exactly once");

        let mut seen: BTreeSet<&str> = BTreeSet::new();
        for desc in &order {
            for dep in reg.dependencies(desc) {
                assert!(
                    seen.contains(dep.as_str()),
                    "{} names {dep}, which has not been emitted yet",
                    desc.name
                );
            }
            seen.insert(desc.name);
        }

        // The ordering has to be stable, not merely correct -- a generator whose output
        // churns between identical runs makes every diff useless.
        let again: Vec<&str> = topological(&reg).iter().map(|d| d.name).collect();
        assert_eq!(order.iter().map(|d| d.name).collect::<Vec<_>>(), again);
    }

    /// The whole module, once. Cheap, and it catches an emitter that produces something
    /// syntactically fine but structurally wrong -- a missing class, a stale reference.
    #[test]
    fn every_registered_type_reaches_the_output() {
        let reg = Registry::read();
        let py = emit_bindings(&reg);

        for desc in reg.types.values() {
            let class = class_name(desc.name);
            let wanted = match desc.kind {
                FfiKind::UnitEnum { .. } => format!("class {class}(enum.IntEnum):"),
                _ => format!("class {class}(ctypes.Structure):"),
            };
            assert!(py.contains(&wanted), "`{}` is registered but not emitted", desc.name);
        }
        for f in &reg.fns {
            assert!(py.contains(&format!("\ndef {}(", f.name)), "{} has no wrapper", f.name);
            assert!(py.contains(&format!("_lib.{}.argtypes", f.name)), "{} unbound", f.name);
        }
        for (name, value) in &reg.consts {
            assert!(py.contains(&format!("{name} = {value}\n")), "{name} is not emitted");
        }
        // No bare `32` where a constant was meant: an array's length is written by name.
        assert!(py.contains("* BOTS_MAX"), "array lengths should be written by name");
        assert!(py.contains("* MAP_SIZE * MAP_SIZE"), "`Map` should keep both dimensions");

        // A handle's free is registered separately from the entry points, so it needs its
        // own check -- it went unbound for a whole session for exactly that reason.
        for h in &reg.handles {
            assert!(py.contains(&format!("_lib.{}.argtypes", h.free)), "{} unbound", h.free);
            assert!(py.contains(&format!("\ndef {}(", h.free)), "{} has no wrapper", h.free);
        }
    }

    /// A generic argument arrives as whatever `mm_ffi_instantiate!` spelled, and `BotId` is
    /// an alias. Annotating it as one gives Python `Optional[ctypes.c_uint8]` for a value
    /// that arrives as a plain `int` -- true of the annotation only, so nothing fails at
    /// runtime and a competitor's editor is the only thing that notices.
    #[test]
    fn a_generic_argument_resolves_through_its_alias() {
        let reg = Registry::read();
        assert_eq!(reg.follow_alias("BotId"), "u8");
        assert_eq!(reg.follow_alias("Team"), "Team", "a registered type is not an alias");
        assert_eq!(reg.follow_alias("f32"), "f32");
        assert_eq!(reg.follow_alias("Vec2"), "Vec2");

        let em = Emitter { reg: &reg, out: String::new() };
        assert_eq!(em.element_annotation("BotId"), "int");
        assert_eq!(em.element_annotation("Team"), "Team");
        assert_eq!(em.element_annotation("Vec2"), "Vec2");
        assert_eq!(em.element_annotation("f32"), "float");
    }

    /// The renaming rule, and the reason it exists: a `_fields_` name is what fixes a
    /// field's offset, so a property can only take over the public spelling by displacing
    /// the raw entry. A rule that stopped firing would leave the property shadowed by the
    /// field and silently never called.
    #[test]
    fn a_field_the_friendly_surface_takes_over_is_renamed() {
        let reg = Registry::read();
        let field = |ty: &str, name: &str| -> String {
            let desc = reg.types[ty];
            let f = fields_of(desc)
                .into_iter()
                .find(|f| f.name == name)
                .unwrap_or_else(|| panic!("{ty} has no field `{name}`"));
            reg.wire_field_name(desc, f)
        };

        // A `StateOption` field, derived from the type.
        assert_eq!(field("SpecialState", "shot"), "_shot");
        assert_eq!(field("SpecialState", "healing"), "_healing");
        assert_eq!(field("SpecialState", "extracting"), "_extracting");
        // A declared one, because `BotArray` being a collection is not in the metadata.
        assert_eq!(field("BotArray", "len"), "_len");
        assert_eq!(field("BotArray", "mask"), "_mask");
        // Everything else keeps its name.
        assert_eq!(field("BotState", "pos"), "pos");
        assert_eq!(field("FleetAction", "rush_order"), "rush_order");
        assert_eq!(field("GameConfig", "map"), "map");

        // And `StateOption`'s own payload is not renamed: `option`/`of` read it directly,
        // and renaming it would break them rather than help anything.
        assert_eq!(field("StateOption<Vec2>", "0"), "f0");
    }

    /// Every type the semantic table names must exist, or its block is emitted nowhere and
    /// the loss is invisible -- the class still works, it is simply missing its methods.
    #[test]
    fn every_declared_semantic_block_has_a_type_to_land_on() {
        let reg = Registry::read();
        for (name, body) in SEMANTIC {
            assert!(reg.types.contains_key(name), "`{name}` has a semantic block but is not registered");
            assert!(body.starts_with('\n'), "`{name}`'s block must open on its own line");
        }
        for (name, _) in PRIVATE_FIELDS {
            assert!(reg.types.contains_key(name), "`{name}` has private fields but is not registered");
        }

        // And the blocks actually reach the output.
        let py = emit_bindings(&reg);
        assert!(py.contains("    def other_team(self) -> Team:"), "Team lost its block");
        assert!(py.contains("    def __len__(self) -> int:"), "BotArray lost its block");
        assert!(py.contains("    def class_(self) -> BotClass:"), "BotState lost its block");
        assert!(py.contains("    def payload_pos(self) -> Vec2:"), "GameState lost its block");
        // The two derived rules, likewise.
        assert!(py.contains("    def shot(self) -> Optional[Vec2]:"));
        assert!(py.contains("    def TargetPosition(cls, pos: Vec2) -> TurnAction:"));
        // `StateOption` gets `Optional`, and *not* the tagged surface: `ffi.md` is explicit
        // that it presents as a plain `Optional[T]` with no wrapper to learn.
        assert!(py.contains("    def of(cls, value: Optional[Vec2]) -> StateOption_Vec2:"));
        assert!(!py.contains("def Some(cls,"), "StateOption should have no Some constructor");
    }
}
