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
    // S,

    // True,  // Later
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
    // PartialAp2(Exp, Box<EvalResult>, Box<EvalResult>),
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
        // "s" => Ok(ParseResult {
        //     exp: Exp::S,
        //     tokens,
        // }),
        x => {
            let num: i64 = x.parse()?;
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
            let left = eval(*left)?;
            let right = eval(*right)?;
            match left {
                // functions which take 1 parameter.
                EvalResult::LeafFunc(Exp::Neg) => match right {
                    EvalResult::Num(n) => Ok(EvalResult::Num(-n)),
                    _ => bail!("can not negate non-numger"),
                },
                // Other functions which take n > 1 parameters.
                EvalResult::LeafFunc(func) => Ok(EvalResult::PartialAp1(func, Box::new(right))),
                // // functions which take 3 parameter. e.g. S-Combinator
                // EvalResult::PartialAp(Exp::S, op1) => {
                //     Ok(EvalResult::PartialAp2(Exp::S, op1, Box::new(right)))
                // }
                EvalResult::PartialAp1(func, op1) => apply_func(func, *op1, right),
                _ => bail!(format!(
                    "Eval error: can not apply app {:?} {:?}",
                    left, right
                )),
            }
        }
        Exp::Add => Ok(EvalResult::LeafFunc(Exp::Add)),
        Exp::Mul => Ok(EvalResult::LeafFunc(Exp::Mul)),
        Exp::Div => Ok(EvalResult::LeafFunc(Exp::Div)),
        Exp::Eq => Ok(EvalResult::LeafFunc(Exp::Eq)),
        Exp::Lt => Ok(EvalResult::LeafFunc(Exp::Lt)),
        Exp::Neg => Ok(EvalResult::LeafFunc(Exp::Neg)),
        // Exp::S => Ok(EvalResult::LeafFunc(Exp::S)),
    }
}

fn apply_func(func: Exp, op1: EvalResult, op2: EvalResult) -> Result<EvalResult> {
    match (func, op1, op2) {
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
        _ => bail!("Eval error: can not apply"),
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

        Ok(())
    }
}
