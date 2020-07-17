extern crate rustc_ast;
extern crate rustc_data_structures;
extern crate rustc_span;
extern crate rustc_target;

use std::mem;

use rustc_ast::ast::{
    AngleBracketedArg, AngleBracketedArgs, AnonConst, Arm, AssocItemKind, AssocTyConstraint,
    AssocTyConstraintKind, Async, AttrId, AttrItem, AttrKind, AttrStyle, Attribute, BareFnTy,
    BinOpKind, BindingMode, Block, BlockCheckMode, BorrowKind, CaptureBy, Const, Crate, CrateSugar,
    Defaultness, EnumDef, Expr, ExprKind, Extern, Field, FieldPat, FloatTy, FnDecl, FnHeader,
    FnRetTy, FnSig, ForeignItemKind, ForeignMod, GenericArg, GenericArgs, GenericBound,
    GenericParam, GenericParamKind, Generics, GlobalAsm, ImplPolarity, InlineAsm, InlineAsmOperand,
    InlineAsmOptions, InlineAsmRegOrRegClass, InlineAsmTemplatePiece, IntTy, IsAuto, Item,
    ItemKind, Label, Lifetime, Lit, LitFloatType, LitIntType, LitKind, LlvmAsmDialect,
    LlvmInlineAsm, LlvmInlineAsmOutput, Local, MacArgs, MacCall, MacDelimiter, MacStmtStyle,
    MacroDef, Mod, Movability, MutTy, Mutability, NodeId, Param, ParenthesizedArgs, Pat, PatKind,
    Path, PathSegment, PolyTraitRef, QSelf, RangeEnd, RangeLimits, RangeSyntax, Stmt, StmtKind,
    StrLit, StrStyle, StructField, TraitBoundModifier, TraitObjectSyntax, TraitRef, Ty, TyKind,
    UintTy, UnOp, Unsafe, UnsafeSource, UseTree, UseTreeKind, Variant, VariantData, VisibilityKind,
    WhereBoundPredicate, WhereClause, WhereEqPredicate, WherePredicate, WhereRegionPredicate,
};
use rustc_ast::ptr::P;
use rustc_ast::token::{self, DelimToken, Token, TokenKind};
use rustc_ast::tokenstream::{DelimSpan, TokenStream, TokenTree};
use rustc_ast::util::comments;
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::thin_vec::ThinVec;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::Ident;
use rustc_span::{sym, Span, Symbol, SyntaxContext, DUMMY_SP};

pub trait SpanlessEq {
    fn eq(&self, other: &Self) -> bool;
}

impl<T: SpanlessEq> SpanlessEq for P<T> {
    fn eq(&self, other: &Self) -> bool {
        SpanlessEq::eq(&**self, &**other)
    }
}

impl<T: SpanlessEq> SpanlessEq for Lrc<T> {
    fn eq(&self, other: &Self) -> bool {
        SpanlessEq::eq(&**self, &**other)
    }
}

impl<T: SpanlessEq> SpanlessEq for Option<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (None, None) => true,
            (Some(this), Some(other)) => SpanlessEq::eq(this, other),
            _ => false,
        }
    }
}

impl<T: SpanlessEq> SpanlessEq for Vec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(a, b)| SpanlessEq::eq(a, b))
    }
}

impl<T: SpanlessEq> SpanlessEq for ThinVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| SpanlessEq::eq(a, b))
    }
}

impl<T: SpanlessEq> SpanlessEq for Spanned<T> {
    fn eq(&self, other: &Self) -> bool {
        SpanlessEq::eq(&self.node, &other.node)
    }
}

impl<A: SpanlessEq, B: SpanlessEq> SpanlessEq for (A, B) {
    fn eq(&self, other: &Self) -> bool {
        SpanlessEq::eq(&self.0, &other.0) && SpanlessEq::eq(&self.1, &other.1)
    }
}

impl<A: SpanlessEq, B: SpanlessEq, C: SpanlessEq> SpanlessEq for (A, B, C) {
    fn eq(&self, other: &Self) -> bool {
        SpanlessEq::eq(&self.0, &other.0)
            && SpanlessEq::eq(&self.1, &other.1)
            && SpanlessEq::eq(&self.2, &other.2)
    }
}

macro_rules! spanless_eq_true {
    ($name:ident) => {
        impl SpanlessEq for $name {
            fn eq(&self, _other: &Self) -> bool {
                true
            }
        }
    };
}

spanless_eq_true!(Span);
spanless_eq_true!(DelimSpan);
spanless_eq_true!(AttrId);
spanless_eq_true!(NodeId);
spanless_eq_true!(SyntaxContext);

macro_rules! spanless_eq_partial_eq {
    ($name:ident) => {
        impl SpanlessEq for $name {
            fn eq(&self, other: &Self) -> bool {
                PartialEq::eq(self, other)
            }
        }
    };
}

spanless_eq_partial_eq!(bool);
spanless_eq_partial_eq!(u8);
spanless_eq_partial_eq!(u16);
spanless_eq_partial_eq!(u128);
spanless_eq_partial_eq!(usize);
spanless_eq_partial_eq!(char);
spanless_eq_partial_eq!(String);
spanless_eq_partial_eq!(Symbol);
spanless_eq_partial_eq!(DelimToken);
spanless_eq_partial_eq!(InlineAsmOptions);

macro_rules! spanless_eq_struct {
    {
        $name:ident $(<$param:ident>)?;
        $([$field:ident $other:ident])*
        $(![$ignore:ident])*
    } => {
        impl $(<$param: SpanlessEq>)* SpanlessEq for $name $(<$param>)* {
            fn eq(&self, other: &Self) -> bool {
                let $name { $($field,)* $($ignore: _,)* } = self;
                let $name { $($field: $other,)* $($ignore: _,)* } = other;
                $(SpanlessEq::eq($field, $other))&&*
            }
        }
    };

    {
        $name:ident $(<$param:ident>)?;
        $([$field:ident $other:ident])*
        $next:ident
        $($rest:ident)*
        $(!$ignore:ident)*
    } => {
        spanless_eq_struct! {
            $name $(<$param>)*;
            $([$field $other])*
            [$next other]
            $($rest)*
            $(!$ignore)*
        }
    };

    {
        $name:ident $(<$param:ident>)?;
        $([$field:ident $other:ident])*
        $(![$ignore:ident])*
        !$next:ident
        $(!$rest:ident)*
    } => {
        spanless_eq_struct! {
            $name $(<$param>)*;
            $([$field $other])*
            $(![$ignore])*
            ![$next]
            $(!$rest)*
        }
    };
}

macro_rules! spanless_eq_enum {
    {
        $name:ident;
        $([$variant:ident $([$field:tt $this:ident $other:ident])*])*
    } => {
        impl SpanlessEq for $name {
            fn eq(&self, other: &Self) -> bool {
                match self {
                    $(
                        $name::$variant { .. } => {}
                    )*
                }
                #[allow(unreachable_patterns)]
                match (self, other) {
                    $(
                        (
                            $name::$variant { $($field: $this),* },
                            $name::$variant { $($field: $other),* },
                        ) => {
                            true $(&& SpanlessEq::eq($this, $other))*
                        }
                    )*
                    _ => false,
                }
            }
        }
    };

    {
        $name:ident;
        $([$variant:ident $($fields:tt)*])*
        $next:ident [$($named:tt)*] ( $i:tt $($field:tt)* )
        $($rest:tt)*
    } => {
        spanless_eq_enum! {
            $name;
            $([$variant $($fields)*])*
            $next [$($named)* [$i this other]] ( $($field)* )
            $($rest)*
        }
    };

    {
        $name:ident;
        $([$variant:ident $($fields:tt)*])*
        $next:ident [$($named:tt)*] ()
        $($rest:tt)*
    } => {
        spanless_eq_enum! {
            $name;
            $([$variant $($fields)*])*
            [$next $($named)*]
            $($rest)*
        }
    };

    {
        $name:ident;
        $([$variant:ident $($fields:tt)*])*
        $next:ident ( $($field:tt)* )
        $($rest:tt)*
    } => {
        spanless_eq_enum! {
            $name;
            $([$variant $($fields)*])*
            $next [] ( $($field)* )
            $($rest)*
        }
    };

    {
        $name:ident;
        $([$variant:ident $($fields:tt)*])*
        $next:ident
        $($rest:tt)*
    } => {
        spanless_eq_enum! {
            $name;
            $([$variant $($fields)*])*
            [$next]
            $($rest)*
        }
    };
}

spanless_eq_struct!(AngleBracketedArgs; span args);
spanless_eq_struct!(AnonConst; id value);
spanless_eq_struct!(Arm; attrs pat guard body span id is_placeholder);
spanless_eq_struct!(AssocTyConstraint; id ident kind span);
spanless_eq_struct!(AttrItem; path args);
spanless_eq_struct!(Attribute; kind id style span);
spanless_eq_struct!(BareFnTy; unsafety ext generic_params decl);
spanless_eq_struct!(Block; stmts id rules span);
spanless_eq_struct!(Crate; module attrs span proc_macros);
spanless_eq_struct!(EnumDef; variants);
spanless_eq_struct!(Expr; id kind span attrs !tokens);
spanless_eq_struct!(Field; attrs id span ident expr is_shorthand is_placeholder);
spanless_eq_struct!(FieldPat; ident pat is_shorthand attrs id span is_placeholder);
spanless_eq_struct!(FnDecl; inputs output);
spanless_eq_struct!(FnHeader; constness asyncness unsafety ext);
spanless_eq_struct!(FnSig; header decl);
spanless_eq_struct!(ForeignMod; abi items);
spanless_eq_struct!(GenericParam; id ident attrs bounds is_placeholder kind);
spanless_eq_struct!(Generics; params where_clause span);
spanless_eq_struct!(GlobalAsm; asm);
spanless_eq_struct!(InlineAsm; template operands options line_spans);
spanless_eq_struct!(Item<K>; attrs id span vis ident kind !tokens);
spanless_eq_struct!(Label; ident);
spanless_eq_struct!(Lifetime; id ident);
spanless_eq_struct!(Lit; token kind span);
spanless_eq_struct!(LlvmInlineAsm; asm asm_str_style outputs inputs clobbers volatile alignstack dialect);
spanless_eq_struct!(LlvmInlineAsmOutput; constraint expr is_rw is_indirect);
spanless_eq_struct!(Local; pat ty init id span attrs);
spanless_eq_struct!(MacCall; path args prior_type_ascription);
spanless_eq_struct!(MacroDef; body macro_rules);
spanless_eq_struct!(Mod; inner items inline);
spanless_eq_struct!(MutTy; ty mutbl);
spanless_eq_struct!(Param; attrs ty pat id span is_placeholder);
spanless_eq_struct!(ParenthesizedArgs; span inputs output);
spanless_eq_struct!(Pat; id kind span);
spanless_eq_struct!(Path; span segments);
spanless_eq_struct!(PathSegment; ident id args);
spanless_eq_struct!(PolyTraitRef; bound_generic_params trait_ref span);
spanless_eq_struct!(QSelf; ty path_span position);
spanless_eq_struct!(Stmt; id kind span);
spanless_eq_struct!(StrLit; style symbol suffix span symbol_unescaped);
spanless_eq_struct!(StructField; attrs id span vis ident ty is_placeholder);
spanless_eq_struct!(Token; kind span);
spanless_eq_struct!(TraitRef; path ref_id);
spanless_eq_struct!(Ty; id kind span);
spanless_eq_struct!(UseTree; prefix kind span);
spanless_eq_struct!(Variant; attrs id span vis ident data disr_expr is_placeholder);
spanless_eq_struct!(WhereBoundPredicate; span bound_generic_params bounded_ty bounds);
spanless_eq_struct!(WhereClause; has_where_token predicates span);
spanless_eq_struct!(WhereEqPredicate; id span lhs_ty rhs_ty);
spanless_eq_struct!(WhereRegionPredicate; span lifetime bounds);
spanless_eq_enum!(AngleBracketedArg; Arg(0) Constraint(0));
spanless_eq_enum!(AssocItemKind; Const(0 1 2) Fn(0 1 2 3) TyAlias(0 1 2 3) MacCall(0));
spanless_eq_enum!(AssocTyConstraintKind; Equality(ty) Bound(bounds));
spanless_eq_enum!(Async; Yes(span closure_id return_impl_trait_id) No);
spanless_eq_enum!(AttrKind; Normal(0) DocComment(0));
spanless_eq_enum!(AttrStyle; Outer Inner);
spanless_eq_enum!(BinOpKind; Add Sub Mul Div Rem And Or BitXor BitAnd BitOr Shl Shr Eq Lt Le Ne Ge Gt);
spanless_eq_enum!(BindingMode; ByRef(0) ByValue(0));
spanless_eq_enum!(BlockCheckMode; Default Unsafe(0));
spanless_eq_enum!(BorrowKind; Ref Raw);
spanless_eq_enum!(CaptureBy; Value Ref);
spanless_eq_enum!(Const; Yes(0) No);
spanless_eq_enum!(CrateSugar; PubCrate JustCrate);
spanless_eq_enum!(Defaultness; Default(0) Final);
spanless_eq_enum!(Extern; None Implicit Explicit(0));
spanless_eq_enum!(FloatTy; F32 F64);
spanless_eq_enum!(FnRetTy; Default(0) Ty(0));
spanless_eq_enum!(ForeignItemKind; Static(0 1 2) Fn(0 1 2 3) TyAlias(0 1 2 3) MacCall(0));
spanless_eq_enum!(GenericArg; Lifetime(0) Type(0) Const(0));
spanless_eq_enum!(GenericArgs; AngleBracketed(0) Parenthesized(0));
spanless_eq_enum!(GenericBound; Trait(0 1) Outlives(0));
spanless_eq_enum!(GenericParamKind; Lifetime Type(default) Const(ty));
spanless_eq_enum!(ImplPolarity; Positive Negative(0));
spanless_eq_enum!(InlineAsmRegOrRegClass; Reg(0) RegClass(0));
spanless_eq_enum!(InlineAsmTemplatePiece; String(0) Placeholder(operand_idx modifier span));
spanless_eq_enum!(IntTy; Isize I8 I16 I32 I64 I128);
spanless_eq_enum!(IsAuto; Yes No);
spanless_eq_enum!(LitFloatType; Suffixed(0) Unsuffixed);
spanless_eq_enum!(LitIntType; Signed(0) Unsigned(0) Unsuffixed);
spanless_eq_enum!(LlvmAsmDialect; Att Intel);
spanless_eq_enum!(MacArgs; Empty Delimited(0 1 2) Eq(0 1));
spanless_eq_enum!(MacDelimiter; Parenthesis Bracket Brace);
spanless_eq_enum!(MacStmtStyle; Semicolon Braces NoBraces);
spanless_eq_enum!(Movability; Static Movable);
spanless_eq_enum!(Mutability; Mut Not);
spanless_eq_enum!(RangeEnd; Included(0) Excluded);
spanless_eq_enum!(RangeLimits; HalfOpen Closed);
spanless_eq_enum!(StmtKind; Local(0) Item(0) Expr(0) Semi(0) Empty MacCall(0));
spanless_eq_enum!(StrStyle; Cooked Raw(0));
spanless_eq_enum!(TokenTree; Token(0) Delimited(0 1 2));
spanless_eq_enum!(TraitBoundModifier; None Maybe MaybeConst MaybeConstMaybe);
spanless_eq_enum!(TraitObjectSyntax; Dyn None);
spanless_eq_enum!(UintTy; Usize U8 U16 U32 U64 U128);
spanless_eq_enum!(UnOp; Deref Not Neg);
spanless_eq_enum!(Unsafe; Yes(0) No);
spanless_eq_enum!(UnsafeSource; CompilerGenerated UserProvided);
spanless_eq_enum!(UseTreeKind; Simple(0 1 2) Nested(0) Glob);
spanless_eq_enum!(VariantData; Struct(0 1) Tuple(0 1) Unit(0));
spanless_eq_enum!(VisibilityKind; Public Crate(0) Restricted(path id) Inherited);
spanless_eq_enum!(WherePredicate; BoundPredicate(0) RegionPredicate(0) EqPredicate(0));
spanless_eq_enum!(ExprKind; Box(0) Array(0) Call(0 1) MethodCall(0 1 2) Tup(0)
    Binary(0 1 2) Unary(0 1) Lit(0) Cast(0 1) Type(0 1) Let(0 1) If(0 1 2)
    While(0 1 2) ForLoop(0 1 2 3) Loop(0 1) Match(0 1) Closure(0 1 2 3 4 5)
    Block(0 1) Async(0 1 2) Await(0) TryBlock(0) Assign(0 1 2) AssignOp(0 1 2)
    Field(0 1) Index(0 1) Range(0 1 2) Path(0 1) AddrOf(0 1 2) Break(0 1)
    Continue(0) Ret(0) InlineAsm(0) LlvmInlineAsm(0) MacCall(0) Struct(0 1 2)
    Repeat(0 1) Paren(0) Try(0) Yield(0) Err);
spanless_eq_enum!(InlineAsmOperand; In(reg expr) Out(reg late expr)
    InOut(reg late expr) SplitInOut(reg late in_expr out_expr) Const(expr)
    Sym(expr));
spanless_eq_enum!(ItemKind; ExternCrate(0) Use(0) Static(0 1 2) Const(0 1 2)
    Fn(0 1 2 3) Mod(0) ForeignMod(0) GlobalAsm(0) TyAlias(0 1 2 3) Enum(0 1)
    Struct(0 1) Union(0 1) Trait(0 1 2 3 4) TraitAlias(0 1)
    Impl(unsafety polarity defaultness constness generics of_trait self_ty items)
    MacCall(0) MacroDef(0));
spanless_eq_enum!(LitKind; Str(0 1) ByteStr(0) Byte(0) Char(0) Int(0 1)
    Float(0 1) Bool(0) Err(0));
spanless_eq_enum!(PatKind; Wild Ident(0 1 2) Struct(0 1 2) TupleStruct(0 1)
    Or(0) Path(0 1) Tuple(0) Box(0) Ref(0 1) Lit(0) Range(0 1 2) Slice(0) Rest
    Paren(0) MacCall(0));
spanless_eq_enum!(TyKind; Slice(0) Array(0 1) Ptr(0) Rptr(0 1) BareFn(0) Never
    Tup(0) Path(0 1) TraitObject(0 1) ImplTrait(0 1) Paren(0) Typeof(0) Infer
    ImplicitSelf MacCall(0) Err CVarArgs);

impl SpanlessEq for Ident {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

// Give up on comparing literals inside of macros because there are so many
// equivalent representations of the same literal; they are tested elsewhere
impl SpanlessEq for token::Lit {
    fn eq(&self, other: &Self) -> bool {
        mem::discriminant(self) == mem::discriminant(other)
    }
}

impl SpanlessEq for RangeSyntax {
    fn eq(&self, _other: &Self) -> bool {
        match self {
            RangeSyntax::DotDotDot | RangeSyntax::DotDotEq => true,
        }
    }
}

impl SpanlessEq for TokenKind {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TokenKind::Literal(this), TokenKind::Literal(other)) => SpanlessEq::eq(this, other),
            (TokenKind::DotDotEq, _) | (TokenKind::DotDotDot, _) => match other {
                TokenKind::DotDotEq | TokenKind::DotDotDot => true,
                _ => false,
            },
            _ => self == other,
        }
    }
}

impl SpanlessEq for TokenStream {
    fn eq(&self, other: &Self) -> bool {
        SpanlessEq::eq(&expand_tts(self), &expand_tts(other))
    }
}

fn expand_tts(tts: &TokenStream) -> Vec<TokenTree> {
    let mut tokens = Vec::new();
    for tt in tts.clone().into_trees() {
        let c = match tt {
            TokenTree::Token(Token {
                kind: TokenKind::DocComment(c),
                ..
            }) => c,
            _ => {
                tokens.push(tt);
                continue;
            }
        };
        let contents = comments::strip_doc_comment_decoration(&c.as_str());
        let style = comments::doc_comment_style(&c.as_str());
        tokens.push(TokenTree::token(TokenKind::Pound, DUMMY_SP));
        if style == AttrStyle::Inner {
            tokens.push(TokenTree::token(TokenKind::Not, DUMMY_SP));
        }
        let lit = token::Lit {
            kind: token::LitKind::Str,
            symbol: Symbol::intern(&contents),
            suffix: None,
        };
        let tts = vec![
            TokenTree::token(TokenKind::Ident(sym::doc, false), DUMMY_SP),
            TokenTree::token(TokenKind::Eq, DUMMY_SP),
            TokenTree::token(TokenKind::Literal(lit), DUMMY_SP),
        ];
        tokens.push(TokenTree::Delimited(
            DelimSpan::dummy(),
            DelimToken::Bracket,
            tts.into_iter().collect::<TokenStream>().into(),
        ));
    }
    tokens
}
