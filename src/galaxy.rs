use crate::prelude::*;
pub use log::*;

fn tokenize(src: &str) -> Vec<&str> {
    src.split_whitespace().collect()
}

fn parse_src(src: &str) -> Result<Exp> {
    let tokens = tokenize(src);
    let parse_result = parse(&tokens)?;
    ensure!(
        parse_result.tokens.is_empty(),
        format!("tokens are not consumed: {:?}", parse_result.tokens)
    );
    Ok(parse_result.exp)
}

struct ParseResult<'a> {
    exp: Exp,
    tokens: &'a [&'a str],
}

#[derive(PartialEq, Clone, Debug)]
pub enum Exp {
    Num(i64),
    Ap(Box<Exp>, Box<Exp>),
    Add,
    Mul,
    Div,
    Eq,
    Lt,
    Neg,
    Inc,
    Dec,
    S,
    C,
    B,
    T, // True
       // False,
}

#[derive(PartialEq, Clone, Debug)]
pub enum EvalResult {
    Num(i64),
    // e.g. "add"
    LeafFunc(Exp),
    // e.g. "ap add 1"
    PartialAp1(Exp, Box<EvalResult>),
    // e.g. "ap ap s x y"
    PartialAp2(Exp, Box<EvalResult>, Box<EvalResult>),
    True,
    False,
}

fn parse<'a>(tokens: &'a [&'a str]) -> Result<ParseResult<'a>> {
    assert!(!tokens.is_empty());
    let (current_token, tokens) = (tokens[0], &tokens[1..]);
    match current_token {
        "ap" => {
            let ParseResult { exp: exp1, tokens } = parse(tokens)?;
            let ParseResult { exp: exp2, tokens } = parse(tokens)?;
            Ok(ParseResult {
                exp: Exp::Ap(Box::new(exp1), Box::new(exp2)),
                tokens,
            })
        }
        "add" => Ok(ParseResult {
            exp: Exp::Add,
            tokens,
        }),
        "eq" => Ok(ParseResult {
            exp: Exp::Eq,
            tokens,
        }),
        "mul" => Ok(ParseResult {
            exp: Exp::Mul,
            tokens,
        }),
        "div" => Ok(ParseResult {
            exp: Exp::Div,
            tokens,
        }),
        "lt" => Ok(ParseResult {
            exp: Exp::Lt,
            tokens,
        }),
        "neg" => Ok(ParseResult {
            exp: Exp::Neg,
            tokens,
        }),
        "inc" => Ok(ParseResult {
            exp: Exp::Inc,
            tokens,
        }),
        "dec" => Ok(ParseResult {
            exp: Exp::Dec,
            tokens,
        }),
        "s" => Ok(ParseResult {
            exp: Exp::S,
            tokens,
        }),
        "c" => Ok(ParseResult {
            exp: Exp::C,
            tokens,
        }),
        "b" => Ok(ParseResult {
            exp: Exp::B,
            tokens,
        }),
        "t" => Ok(ParseResult {
            exp: Exp::T,
            tokens,
        }),
        x => {
            // TODO: Add context error message.
            let num: i64 = x.parse().context("number parse error")?;
            Ok(ParseResult {
                exp: Exp::Num(num),
                tokens,
            })
        }
    }
}

fn eval(exp: Exp) -> Result<EvalResult> {
    match exp {
        Exp::Num(n) => Ok(EvalResult::Num(n)),
        Exp::Ap(left, right) => {
            // TODO: Don't eval right hand if we don't use it.
            // e.g. app app t 1 long-expression  => we don't need to eval long-expression
            let left = eval(*left)?;
            let right = eval(*right)?;
            apply(left, right)
        }
        Exp::Add => Ok(EvalResult::LeafFunc(Exp::Add)),
        Exp::Mul => Ok(EvalResult::LeafFunc(Exp::Mul)),
        Exp::Div => Ok(EvalResult::LeafFunc(Exp::Div)),
        Exp::Eq => Ok(EvalResult::LeafFunc(Exp::Eq)),
        Exp::Lt => Ok(EvalResult::LeafFunc(Exp::Lt)),
        Exp::Neg => Ok(EvalResult::LeafFunc(Exp::Neg)),
        Exp::Inc => Ok(EvalResult::LeafFunc(Exp::Inc)),
        Exp::Dec => Ok(EvalResult::LeafFunc(Exp::Dec)),
        Exp::S => Ok(EvalResult::LeafFunc(Exp::S)),
        Exp::C => Ok(EvalResult::LeafFunc(Exp::C)),
        Exp::B => Ok(EvalResult::LeafFunc(Exp::B)),
        Exp::T => Ok(EvalResult::LeafFunc(Exp::T)),
    }
}

fn apply(f: EvalResult, x0: EvalResult) -> Result<EvalResult> {
    match f {
        EvalResult::LeafFunc(exp) => match (exp, x0) {
            (Exp::Neg, EvalResult::Num(n)) => Ok(EvalResult::Num(-n)),
            (Exp::Neg, _) => bail!("can not apply"),
            (Exp::Inc, EvalResult::Num(n)) => Ok(EvalResult::Num(n + 1)),
            (Exp::Inc, _) => bail!("can not apply"),
            (Exp::Dec, EvalResult::Num(n)) => Ok(EvalResult::Num(n - 1)),
            (Exp::Dec, _) => bail!("can not apply"),
            (exp, x0) => Ok(EvalResult::PartialAp1(exp, Box::new(x0))),
        },
        EvalResult::PartialAp1(exp, op0) => match (exp, *op0, x0) {
            (Exp::Add, EvalResult::Num(n1), EvalResult::Num(n2)) => Ok(EvalResult::Num(n1 + n2)),
            (Exp::Mul, EvalResult::Num(n1), EvalResult::Num(n2)) => Ok(EvalResult::Num(n1 * n2)),
            (Exp::Div, EvalResult::Num(n1), EvalResult::Num(n2)) => Ok(EvalResult::Num(n1 / n2)),
            (Exp::Eq, EvalResult::Num(n1), EvalResult::Num(n2)) => {
                if n1 == n2 {
                    Ok(EvalResult::True)
                } else {
                    Ok(EvalResult::False)
                }
            }
            (Exp::Lt, EvalResult::Num(n1), EvalResult::Num(n2)) => {
                if n1 < n2 {
                    Ok(EvalResult::True)
                } else {
                    Ok(EvalResult::False)
                }
            }
            (Exp::S, x0, x1) => Ok(EvalResult::PartialAp2(Exp::S, Box::new(x0), Box::new(x1))),
            (Exp::C, x0, x1) => Ok(EvalResult::PartialAp2(Exp::C, Box::new(x0), Box::new(x1))),
            (Exp::B, x0, x1) => Ok(EvalResult::PartialAp2(Exp::B, Box::new(x0), Box::new(x1))),
            (Exp::T, x0, _) => Ok(x0),
            (exp, x0, x1) => bail!("can not apply: exp: {:?}, x0: {:?}, x1: {:?}", exp, x0, x1),
        },
        EvalResult::PartialAp2(exp, op0, op1) => match (exp, *op0, *op1, x0) {
            (Exp::S, x0, x1, x2) => {
                // ap ap ap s x0 x1 x2   =   ap ap x0 x2 ap x1 x2
                // ap ap ap s add inc 1   =   3

                // TODO: Avoid clone. Use Rc?
                let ap_x0_x2 = apply(x0, x2.clone())?;
                let ap_x1_x2 = apply(x1, x2)?;
                apply(ap_x0_x2, ap_x1_x2)
            }
            (Exp::C, x0, x1, x2) => {
                // ap ap ap c x0 x1 x2   =   ap ap x0 x2 x1
                // ap ap ap c add 1 2   =   3
                apply(apply(x0, x2)?, x1)
            }
            (Exp::B, x0, x1, x2) => {
                // ap ap ap b x0 x1 x2   =   ap x0 ap x1 x2
                // ap ap ap b inc dec x0   =   x0
                apply(x0, apply(x1, x2)?)
            }
            _ => bail!("can not apply"),
        },
        _ => {
            bail!("can not apply");
        }
    }
}

pub fn eval_src(src: &str) -> Result<EvalResult> {
    let exp = parse_src(src)?;
    eval(exp)
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn tokenize_test() {
        assert_eq!(tokenize("ap ap add 1 2"), &["ap", "ap", "add", "1", "2"]);
        assert_eq!(
            tokenize(" ap ap add 1   2  "),
            &["ap", "ap", "add", "1", "2"]
        );
    }

    #[test]
    fn parse_test() -> Result<()> {
        assert_eq!(parse_src("1")?, Exp::Num(1));
        assert_eq!(parse_src("add")?, Exp::Add);
        assert_eq!(
            parse_src("ap ap add 1 2")?,
            Exp::Ap(
                Box::new(Exp::Ap(Box::new(Exp::Add), Box::new(Exp::Num(1)))),
                Box::new(Exp::Num(2))
            )
        );
        assert_eq!(
            parse_src("ap ap eq 1 2")?,
            Exp::Ap(
                Box::new(Exp::Ap(Box::new(Exp::Eq), Box::new(Exp::Num(1)))),
                Box::new(Exp::Num(2))
            )
        );
        assert!(parse_src("add 1").is_err());
        Ok(())
    }

    #[test]
    fn eval_test() -> Result<()> {
        // add
        assert_eq!(eval_src("ap ap add 1 2")?, EvalResult::Num(3));
        assert_eq!(eval_src("ap ap add 3 ap ap add 1 2")?, EvalResult::Num(6));

        // eq
        assert_eq!(eval_src("ap ap eq 1 1")?, EvalResult::True);
        assert_eq!(eval_src("ap ap eq 1 2")?, EvalResult::False);

        // mul
        assert_eq!(eval_src("ap ap mul 2 4")?, EvalResult::Num(8));
        assert_eq!(eval_src("ap ap add 3 ap ap mul 2 4")?, EvalResult::Num(11));

        // div
        assert_eq!(eval_src("ap ap div 4 2")?, EvalResult::Num(2));
        assert_eq!(eval_src("ap ap div 4 3")?, EvalResult::Num(1));
        assert_eq!(eval_src("ap ap div 4 4")?, EvalResult::Num(1));
        assert_eq!(eval_src("ap ap div 4 5")?, EvalResult::Num(0));
        assert_eq!(eval_src("ap ap div 5 2")?, EvalResult::Num(2));
        assert_eq!(eval_src("ap ap div 6 -2")?, EvalResult::Num(-3));
        assert_eq!(eval_src("ap ap div 5 -3")?, EvalResult::Num(-1));
        assert_eq!(eval_src("ap ap div -5 3")?, EvalResult::Num(-1));
        assert_eq!(eval_src("ap ap div -5 -3")?, EvalResult::Num(1));

        // lt
        assert_eq!(eval_src("ap ap lt 0 -1")?, EvalResult::False);
        assert_eq!(eval_src("ap ap lt 0 0")?, EvalResult::False);
        assert_eq!(eval_src("ap ap lt 0 1")?, EvalResult::True);

        // neg
        assert_eq!(eval_src("ap neg 0")?, EvalResult::Num(0));
        assert_eq!(eval_src("ap neg 1")?, EvalResult::Num(-1));
        assert_eq!(eval_src("ap neg -1")?, EvalResult::Num(1));
        assert_eq!(eval_src("ap ap add ap neg 1 2")?, EvalResult::Num(1));

        // inc
        assert_eq!(eval_src("ap inc 0")?, EvalResult::Num(1));
        assert_eq!(eval_src("ap inc 1")?, EvalResult::Num(2));

        // dec
        assert_eq!(eval_src("ap dec 0")?, EvalResult::Num(-1));
        assert_eq!(eval_src("ap dec 1")?, EvalResult::Num(0));

        // s
        // assert_eq!(eval_src("ap ap ap s add inc 1", EvalResult::Num(3));  // inc is not implemented yet.
        assert_eq!(eval_src("ap ap ap s mul ap add 1 6")?, EvalResult::Num(42));

        // c
        assert_eq!(eval_src("ap ap ap c add 1 2")?, EvalResult::Num(3));

        // b
        // ap ap ap b x0 x1 x2   =   ap x0 ap x1 x2
        // ap ap ap b inc dec x0   =   x0
        assert_eq!(eval_src("ap ap ap b neg neg 1")?, EvalResult::Num(1));

        // t
        // ap ap t x0 x1   =   x0
        // ap ap t 1 5   =   1
        // ap ap t t i   =   t
        // ap ap t t ap inc 5   =   t
        // ap ap t ap inc 5 t   =   6
        assert_eq!(eval_src("ap ap t 1 5")?, EvalResult::Num(1));
        assert_eq!(eval_src("ap ap t t 1")?, EvalResult::LeafFunc(Exp::T));
        assert_eq!(
            eval_src("ap ap t t ap inc 5")?,
            EvalResult::LeafFunc(Exp::T)
        );
        assert_eq!(eval_src("ap ap t ap inc 5 t")?, EvalResult::Num(6));

        Ok(())
    }
}
