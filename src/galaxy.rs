use crate::prelude::*;
pub use log::*;

use regex::Regex;

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
    T,
    F,
    I,
    Cons,
    Car,
    Cdr,
    Nil,
    Isnil,
    Var(u64),
}

#[derive(PartialEq, Clone, Debug)]
pub enum EvalResult {
    Num(i64),
    Func(Exp),
    PartialAp1(Exp, Box<EvalResult>),
    PartialAp2(Exp, Box<EvalResult>, Box<EvalResult>),
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
        "f" => Ok(ParseResult {
            exp: Exp::F,
            tokens,
        }),
        "i" => Ok(ParseResult {
            exp: Exp::I,
            tokens,
        }),
        "cons" => Ok(ParseResult {
            exp: Exp::Cons,
            tokens,
        }),
        "car" => Ok(ParseResult {
            exp: Exp::Car,
            tokens,
        }),
        "cdr" => Ok(ParseResult {
            exp: Exp::Cdr,
            tokens,
        }),
        "nil" => Ok(ParseResult {
            exp: Exp::Nil,
            tokens,
        }),
        "isnil" => Ok(ParseResult {
            exp: Exp::Isnil,
            tokens,
        }),
        x => {
            if x.as_bytes()[0] == b':' {
                let var_id: u64 = x[1..].parse()?;
                Ok(ParseResult {
                    exp: Exp::Var(var_id),
                    tokens,
                })
            } else {
                // TODO: Add context error message.
                let num: i64 = x.parse().context("number parse error")?;
                Ok(ParseResult {
                    exp: Exp::Num(num),
                    tokens,
                })
            }
        }
    }
}

struct Galaxy {
    galaxy_id: u64,
    variables: HashMap<u64, Exp>,
    results: HashMap<u64, EvalResult>,
}

impl Galaxy {
    fn new_for_test(src: &str) -> Result<Galaxy> {
        let exp = parse_src(src)?;
        Ok(Galaxy {
            galaxy_id: 1,
            variables: {
                let mut map = HashMap::new();
                map.insert(1, exp);
                map
            },
            results: HashMap::new(),
        })
    }

    fn eval_galaxy(&mut self) -> Result<EvalResult> {
        self.eval(self.variables[&self.galaxy_id].clone())
    }

    fn new(src: &str) -> Result<Galaxy> {
        let lines = src.trim().split('\n').collect::<Vec<_>>();
        println!("last line: {}", lines[lines.len() - 1]);

        // assert_eq!(lines.len(), 393);

        let galaxy_line_re = Regex::new(r"galaxy *= :*(\d+)$").unwrap();
        let cap = galaxy_line_re.captures(lines[lines.len() - 1]).unwrap();
        let galaxy_id: u64 = cap[1].parse()?;
        println!("galaxy_id: {}", galaxy_id);

        Ok(Galaxy {
            galaxy_id,
            variables: {
                let mut map = HashMap::new();
                let re = Regex::new(r":(\d+) *= *(.*)$").unwrap();

                for line in lines.iter().take(lines.len() - 1) {
                    let cap = re.captures(line).unwrap();
                    map.insert(cap[1].parse::<u64>()?, parse_src(&cap[2])?);
                    // println!("{}, {}", &cap[1], &cap[2]);
                }
                map
            },
            results: HashMap::new(),
        })
    }

    fn eval(&mut self, exp: Exp) -> Result<EvalResult> {
        // println!("eval: {:?}", exp);
        match exp {
            Exp::Num(n) => Ok(EvalResult::Num(n)),
            Exp::Ap(left, right) => {
                // TODO: Don't eval right hand if we don't use it.
                // e.g. app app t 1 long-expression  => we don't need to eval long-expression
                let left = self.eval(*left)?;
                let right = self.eval(*right)?;
                self.apply(left, right)
            }
            Exp::Add => Ok(EvalResult::Func(Exp::Add)),
            Exp::Mul => Ok(EvalResult::Func(Exp::Mul)),
            Exp::Div => Ok(EvalResult::Func(Exp::Div)),
            Exp::Eq => Ok(EvalResult::Func(Exp::Eq)),
            Exp::Lt => Ok(EvalResult::Func(Exp::Lt)),
            Exp::Neg => Ok(EvalResult::Func(Exp::Neg)),
            Exp::Inc => Ok(EvalResult::Func(Exp::Inc)),
            Exp::Dec => Ok(EvalResult::Func(Exp::Dec)),
            Exp::S => Ok(EvalResult::Func(Exp::S)),
            Exp::C => Ok(EvalResult::Func(Exp::C)),
            Exp::B => Ok(EvalResult::Func(Exp::B)),
            Exp::T => Ok(EvalResult::Func(Exp::T)),
            Exp::F => Ok(EvalResult::Func(Exp::F)),
            Exp::I => Ok(EvalResult::Func(Exp::I)),
            Exp::Cons => Ok(EvalResult::Func(Exp::Cons)),
            Exp::Car => Ok(EvalResult::Func(Exp::Car)),
            Exp::Cdr => Ok(EvalResult::Func(Exp::Cdr)),
            Exp::Nil => Ok(EvalResult::Func(Exp::Nil)),
            Exp::Isnil => Ok(EvalResult::Func(Exp::Isnil)),
            Exp::Var(n) => self.eval_var(n),
        }
    }

    fn eval_var(&mut self, variable_id: u64) -> Result<EvalResult> {
        if let Some(result) = self.results.get(&variable_id) {
            Ok(result.clone())
        } else {
            let result = self.eval(self.variables[&variable_id].clone())?;
            self.results.insert(variable_id, result.clone());
            Ok(result)
        }
    }

    fn apply(&mut self, f: EvalResult, x0: EvalResult) -> Result<EvalResult> {
        // println!("apply: f: {:?}, x0: {:?}", f, x0);
        match f {
            EvalResult::Func(exp) => match (exp, x0) {
                (Exp::Neg, EvalResult::Num(n)) => Ok(EvalResult::Num(-n)),
                (Exp::Neg, _) => bail!("can not apply"),
                (Exp::Inc, EvalResult::Num(n)) => Ok(EvalResult::Num(n + 1)),
                (Exp::Inc, _) => bail!("can not apply"),
                (Exp::Dec, EvalResult::Num(n)) => Ok(EvalResult::Num(n - 1)),
                (Exp::Dec, _) => bail!("can not apply"),
                (Exp::I, x0) => Ok(x0),
                // ap car x2 = ap x2 t
                (Exp::Car, x0) => self.apply(x0, EvalResult::Func(Exp::T)),
                (Exp::Cdr, x0) => self.apply(x0, EvalResult::Func(Exp::F)),
                (Exp::Nil, _) => Ok(EvalResult::Func(Exp::T)),
                (Exp::Isnil, EvalResult::Func(Exp::Nil)) => Ok(EvalResult::Func(Exp::T)),
                (Exp::Isnil, _) => Ok(EvalResult::Func(Exp::F)),
                (exp, x0) => Ok(EvalResult::PartialAp1(exp, Box::new(x0))),
            },
            EvalResult::PartialAp1(exp, op0) => match (exp, *op0, x0) {
                (Exp::Add, EvalResult::Num(n1), EvalResult::Num(n2)) => {
                    Ok(EvalResult::Num(n1 + n2))
                }
                (Exp::Mul, EvalResult::Num(n1), EvalResult::Num(n2)) => {
                    Ok(EvalResult::Num(n1 * n2))
                }
                (Exp::Div, EvalResult::Num(n1), EvalResult::Num(n2)) => {
                    Ok(EvalResult::Num(n1 / n2))
                }
                (Exp::Eq, EvalResult::Num(n1), EvalResult::Num(n2)) => {
                    if n1 == n2 {
                        Ok(EvalResult::Func(Exp::T))
                    } else {
                        Ok(EvalResult::Func(Exp::F))
                    }
                }
                (Exp::Lt, EvalResult::Num(n1), EvalResult::Num(n2)) => {
                    if n1 < n2 {
                        Ok(EvalResult::Func(Exp::T))
                    } else {
                        Ok(EvalResult::Func(Exp::F))
                    }
                }
                (Exp::S, x0, x1) => Ok(EvalResult::PartialAp2(Exp::S, Box::new(x0), Box::new(x1))),
                (Exp::C, x0, x1) => Ok(EvalResult::PartialAp2(Exp::C, Box::new(x0), Box::new(x1))),
                (Exp::B, x0, x1) => Ok(EvalResult::PartialAp2(Exp::B, Box::new(x0), Box::new(x1))),
                (Exp::T, x0, _) => Ok(x0),
                (Exp::F, _, x1) => Ok(x1),
                (Exp::Cons, x0, x1) => Ok(EvalResult::PartialAp2(
                    Exp::Cons,
                    Box::new(x0),
                    Box::new(x1),
                )),
                (exp, x0, x1) => bail!("can not apply: exp: {:?}, x0: {:?}, x1: {:?}", exp, x0, x1),
            },
            EvalResult::PartialAp2(exp, op0, op1) => match (exp, *op0, *op1, x0) {
                (Exp::S, x0, x1, x2) => {
                    // ap ap ap s x0 x1 x2   =   ap ap x0 x2 ap x1 x2
                    // ap ap ap s add inc 1   =   3

                    // TODO: Avoid clone. Use Rc?
                    let ap_x0_x2 = self.apply(x0, x2.clone())?;
                    let ap_x1_x2 = self.apply(x1, x2)?;
                    self.apply(ap_x0_x2, ap_x1_x2)
                }
                (Exp::C, x0, x1, x2) => {
                    // ap ap ap c x0 x1 x2   =   ap ap x0 x2 x1
                    // ap ap ap c add 1 2   =   3
                    let r1 = self.apply(x0, x2)?;
                    self.apply(r1, x1)
                }
                (Exp::B, x0, x1, x2) => {
                    // ap ap ap b x0 x1 x2   =   ap x0 ap x1 x2
                    // ap ap ap b inc dec x0   =   x0
                    let r1 = self.apply(x1, x2)?;
                    self.apply(x0, r1)
                }
                (Exp::Cons, x0, x1, x2) => {
                    // cons
                    // ap ap ap cons x0 x1 x2   =   ap ap x2 x0 x1
                    let r1 = self.apply(x2, x0)?;
                    self.apply(r1, x1)
                }
                _ => bail!("can not apply"),
            },
            x => {
                bail!("can not apply: {:?}", x);
            }
        }
    }
}

pub fn eval_src(src: &str) -> Result<EvalResult> {
    // let exp = parse_src(src)?;
    // eval(exp)
    let mut galaxy = Galaxy::new_for_test(src)?;
    galaxy.eval_galaxy()
}

pub fn run_galaxy(src: &str) -> Result<()> {
    let _galaxy = Galaxy::new(src)?;

    // let initial_state = EvalResult::Func(Expr::Nil);
    // let initial_point = todo!(); // (0, 0)

    Ok(())
}

// fn send(s: &str) -> String {
//     todo!()
// }

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
        let t = EvalResult::Func(Exp::T);
        let f = EvalResult::Func(Exp::F);

        // add
        assert_eq!(eval_src("ap ap add 1 2")?, EvalResult::Num(3));
        assert_eq!(eval_src("ap ap add 3 ap ap add 1 2")?, EvalResult::Num(6));

        // eq
        assert_eq!(eval_src("ap ap eq 1 1")?, t);
        assert_eq!(eval_src("ap ap eq 1 2")?, f);

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
        assert_eq!(eval_src("ap ap lt 0 -1")?, f);
        assert_eq!(eval_src("ap ap lt 0 0")?, f);
        assert_eq!(eval_src("ap ap lt 0 1")?, t);

        Ok(())
    }

    #[test]
    fn eval_unary_test() -> Result<()> {
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

        Ok(())
    }

    #[test]
    fn eval_combinator_test() -> Result<()> {
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
        assert_eq!(eval_src("ap ap t t 1")?, EvalResult::Func(Exp::T));
        assert_eq!(eval_src("ap ap t t ap inc 5")?, EvalResult::Func(Exp::T));
        assert_eq!(eval_src("ap ap t ap inc 5 t")?, EvalResult::Num(6));

        // f
        assert_eq!(eval_src("ap ap f 1 2")?, EvalResult::Num(2));

        // i
        assert_eq!(eval_src("ap i 0")?, EvalResult::Num(0));
        assert_eq!(eval_src("ap i i")?, EvalResult::Func(Exp::I));

        Ok(())
    }

    #[test]
    fn eval_cons_test() -> Result<()> {
        // car, cdr, cons
        // car
        // ap car ap ap cons x0 x1   =   x0
        // ap car x2   =   ap x2 t
        assert_eq!(eval_src("ap car ap ap cons 0 1")?, EvalResult::Num(0));
        assert_eq!(eval_src("ap cdr ap ap cons 0 1")?, EvalResult::Num(1));

        // nil
        // ap nil x0   =   t
        assert_eq!(eval_src("ap nil 1")?, EvalResult::Func(Exp::T));

        // isnil
        assert_eq!(eval_src("ap isnil nil")?, EvalResult::Func(Exp::T));
        assert_eq!(eval_src("ap isnil 1")?, EvalResult::Func(Exp::F));

        Ok(())
    }

    #[ignore]
    #[test]
    fn eval_galaxy_test() -> Result<()> {
        assert_eq!(
            eval_src("ap ap cons 7 ap ap cons 123229502148636 nil")?,
            EvalResult::Func(Exp::T)
        );
        Ok(())
    }

    #[ignore]
    #[test]
    fn run_galaxy_test() -> Result<()> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("task/galaxy.txt");
        // path.push("task/a.txt");
        let src = std::fs::read_to_string(path)?.trim().to_string();

        run_galaxy(&src)
    }
}
