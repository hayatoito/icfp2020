#![allow(dead_code)]

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

struct Galaxy {
    src: String,
}

struct ParseResult<'a> {
    exp: Exp,
    tokens: &'a [&'a str],
}

fn parse<'a>(tokens: &'a [&'a str]) -> Result<ParseResult<'a>> {
    assert!(!tokens.is_empty());
    match tokens[0] {
        "ap" => {
            let ParseResult { exp: exp1, tokens } = parse(&tokens[1..])?;
            let ParseResult { exp: exp2, tokens } = parse(tokens)?;
            Ok(ParseResult {
                exp: Exp::Ap(Box::new(exp1), Box::new(exp2)),
                tokens,
            })
        }
        "add" => Ok(ParseResult {
            exp: Exp::Add,
            tokens: &tokens[1..],
        }),
        x => {
            let num: i64 = x.parse()?;
            Ok(ParseResult {
                exp: Exp::Num(num),
                tokens: &tokens[1..],
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
                EvalResult::LeafFunc(func) => Ok(EvalResult::PartialAp(
                    Box::new(EvalResult::LeafFunc(func)),
                    Box::new(right),
                )),
                EvalResult::PartialAp(func, op1) => apply_func(*func, *op1, right),
                _ => bail!(format!(
                    "Eval error: can not apply app {:?} {:?}",
                    left, right
                )),
            }
        }
        Exp::Add => Ok(EvalResult::LeafFunc(Exp::Add)),
    }
}

fn apply_func(func: EvalResult, op1: EvalResult, op2: EvalResult) -> Result<EvalResult> {
    match func {
        EvalResult::LeafFunc(func) => match (func, op1, op2) {
            (Exp::Add, EvalResult::Num(n1), EvalResult::Num(n2)) => Ok(EvalResult::Num(n1 + n2)),
            _ => bail!("Eval error: can not apply"),
        },
        _ => bail!(format!(
            "Eval error: can not apply {:?} {:?} {:?}",
            func, op1, op2
        )),
    }
}

fn eval_src(src: &str) -> Result<EvalResult> {
    let exp = parse_src(src)?;
    eval(exp)
}

#[derive(PartialEq, Clone, Debug)]
enum Exp {
    Num(i64),
    Ap(Box<Exp>, Box<Exp>),
    Add,
}

#[derive(PartialEq, Clone, Debug)]
enum EvalResult {
    Num(i64),
    // e.g. "add"
    LeafFunc(Exp),
    // e.g. "ap add 1"
    PartialAp(Box<EvalResult>, Box<EvalResult>),
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
        assert!(parse_src("add 1").is_err());
        Ok(())
    }

    #[test]
    fn eval_test() -> Result<()> {
        assert_eq!(eval_src("ap ap add 1 2")?, EvalResult::Num(3));
        assert_eq!(eval_src("ap ap add 3 ap ap add 1 2")?, EvalResult::Num(6));
        Ok(())
    }
}
