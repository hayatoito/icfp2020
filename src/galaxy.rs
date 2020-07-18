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
pub enum Value {
    Num(i64),
    Func(Exp),
    PartialAp1(Exp, Exp),
    PartialAp2(Exp, Exp, Exp),
}

fn parse<'a>(tokens: &'a [&'a str]) -> Result<ParseResult<'a>> {
    assert!(!tokens.is_empty());
    let (current_token, tokens) = (tokens[0], &tokens[1..]);
    match current_token {
        "ap" => {
            // TODO: parse overflows in debug build.
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
    results: HashMap<u64, Value>,
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

    fn eval_galaxy(&mut self) -> Result<Value> {
        self.eval(self.variables[&self.galaxy_id].clone())
    }

    fn new(src: &str) -> Result<Galaxy> {
        let lines = src.trim().split('\n').collect::<Vec<_>>();
        println!("last line: {}", lines[lines.len() - 1]);
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
                    println!("parse: line: {}", line);
                    let cap = re.captures(line).unwrap();
                    map.insert(cap[1].parse::<u64>()?, parse_src(&cap[2])?);
                    // println!("{}, {}", &cap[1], &cap[2]);
                }
                map
            },
            results: HashMap::new(),
        })
    }

    fn eval(&mut self, exp: Exp) -> Result<Value> {
        println!("eval: {:?}", exp);
        match exp {
            Exp::Num(n) => Ok(Value::Num(n)),
            Exp::Ap(left, right) => self.apply(*left, *right),
            Exp::Add => Ok(Value::Func(Exp::Add)),
            Exp::Mul => Ok(Value::Func(Exp::Mul)),
            Exp::Div => Ok(Value::Func(Exp::Div)),
            Exp::Eq => Ok(Value::Func(Exp::Eq)),
            Exp::Lt => Ok(Value::Func(Exp::Lt)),
            Exp::Neg => Ok(Value::Func(Exp::Neg)),
            Exp::Inc => Ok(Value::Func(Exp::Inc)),
            Exp::Dec => Ok(Value::Func(Exp::Dec)),
            Exp::S => Ok(Value::Func(Exp::S)),
            Exp::C => Ok(Value::Func(Exp::C)),
            Exp::B => Ok(Value::Func(Exp::B)),
            Exp::T => Ok(Value::Func(Exp::T)),
            Exp::F => Ok(Value::Func(Exp::F)),
            Exp::I => Ok(Value::Func(Exp::I)),
            Exp::Cons => Ok(Value::Func(Exp::Cons)),
            Exp::Car => Ok(Value::Func(Exp::Car)),
            Exp::Cdr => Ok(Value::Func(Exp::Cdr)),
            Exp::Nil => Ok(Value::Func(Exp::Nil)),
            Exp::Isnil => Ok(Value::Func(Exp::Isnil)),
            Exp::Var(n) => self.eval_var(n),
        }
    }

    fn eval_var(&mut self, variable_id: u64) -> Result<Value> {
        if let Some(result) = self.results.get(&variable_id) {
            Ok(result.clone())
        } else {
            let result = self.eval(self.variables[&variable_id].clone())?;
            self.results.insert(variable_id, result.clone());
            Ok(result)
        }
    }

    // fn apply(&mut self, f: Value, x0: Value) -> Result<Value> {

    fn apply(&mut self, f: Exp, x0: Exp) -> Result<Value> {
        // println!("apply: f: {:?}, x0: {:?}", f, x0);
        let f = self.eval(f)?;
        match f {
            Value::Func(exp) => match exp {
                Exp::Neg => match self.eval(x0)? {
                    Value::Num(n) => Ok(Value::Num(-n)),
                    _ => bail!("can not apply"),
                },
                Exp::Inc => match self.eval(x0)? {
                    Value::Num(n) => Ok(Value::Num(n + 1)),
                    _ => bail!("can not apply"),
                },
                Exp::Dec => match self.eval(x0)? {
                    Value::Num(n) => Ok(Value::Num(n - 1)),
                    _ => bail!("can not apply"),
                },
                Exp::I => self.eval(x0),
                // ap car x2 = ap x2 t
                Exp::Car => self.apply(x0, Exp::T),
                Exp::Cdr => self.apply(x0, Exp::F),
                Exp::Nil => Ok(Value::Func(Exp::T)),
                Exp::Isnil => match self.eval(x0)? {
                    Value::Func(Exp::Nil) => Ok(Value::Func(Exp::T)),
                    _ => Ok(Value::Func(Exp::F)),
                },
                exp => Ok(Value::PartialAp1(exp, x0)),
            },
            Value::PartialAp1(exp, e0) => {
                let e1 = x0; // For readability.
                match exp {
                    Exp::Add => match (self.eval(e0)?, self.eval(e1)?) {
                        (Value::Num(n0), Value::Num(n1)) => Ok(Value::Num(n0 + n1)),
                        _ => bail!("can not apply"),
                    },
                    Exp::Mul => match (self.eval(e0)?, self.eval(e1)?) {
                        (Value::Num(n0), Value::Num(n1)) => Ok(Value::Num(n0 * n1)),
                        _ => bail!("can not apply"),
                    },
                    Exp::Div => match (self.eval(e0)?, self.eval(e1)?) {
                        (Value::Num(n0), Value::Num(n1)) => Ok(Value::Num(n0 / n1)),
                        _ => bail!("can not apply"),
                    },
                    Exp::Eq => match (self.eval(e0)?, self.eval(e1)?) {
                        (Value::Num(n0), Value::Num(n1)) => {
                            if n0 == n1 {
                                Ok(Value::Func(Exp::T))
                            } else {
                                Ok(Value::Func(Exp::F))
                            }
                        }
                        _ => bail!("can not apply"),
                    },
                    Exp::Lt => match (self.eval(e0)?, self.eval(e1)?) {
                        (Value::Num(n0), Value::Num(n1)) => {
                            if n0 < n1 {
                                Ok(Value::Func(Exp::T))
                            } else {
                                Ok(Value::Func(Exp::F))
                            }
                        }
                        _ => bail!("can not apply"),
                    },
                    Exp::S => Ok(Value::PartialAp2(Exp::S, e0, e1)),
                    Exp::C => Ok(Value::PartialAp2(Exp::C, e0, e1)),
                    Exp::B => Ok(Value::PartialAp2(Exp::B, e0, e1)),
                    Exp::T => self.eval(e0),
                    // New
                    Exp::F => self.eval(e1),
                    Exp::Cons => Ok(Value::PartialAp2(Exp::Cons, e0, e1)),
                    exp => bail!("can not apply: exp: {:?}, e0: {:?}, e1: {:?}", exp, e0, e1),
                }
            }
            Value::PartialAp2(exp, e0, e1) => {
                let e2 = x0;
                match exp {
                    Exp::S => {
                        // ap ap ap s x0 x1 x2   =   ap ap x0 x2 ap x1 x2
                        // ap ap ap s add inc 1   =   3

                        // TODO: Avoid clone. Use Rc?
                        let ap_x0_x2 = Exp::Ap(Box::new(e0), Box::new(e2.clone()));
                        let ap_x1_x2 = Exp::Ap(Box::new(e1), Box::new(e2));
                        self.apply(ap_x0_x2, ap_x1_x2)
                    }
                    Exp::C => {
                        // ap ap ap c x0 x1 x2   =   ap ap x0 x2 x1
                        // ap ap ap c add 1 2   =   3
                        self.apply(Exp::Ap(Box::new(e0), Box::new(e2)), e1)
                    }
                    Exp::B => {
                        // ap ap ap b x0 x1 x2   =   ap x0 ap x1 x2
                        // ap ap ap b inc dec x0   =   x0
                        self.apply(e0, Exp::Ap(Box::new(e1), Box::new(e2)))
                    }
                    Exp::Cons => {
                        // cons
                        // ap ap ap cons x0 x1 x2   =   ap ap x2 x0 x1
                        self.apply(Exp::Ap(Box::new(e2), Box::new(e0)), e1)
                    }
                    _ => bail!("can not apply"),
                }
            }
            Value::Num(_) => bail!("can not apply"),
        }
    }
}

pub fn eval_src(src: &str) -> Result<Value> {
    // let exp = parse_src(src)?;
    // eval(exp)
    let mut galaxy = Galaxy::new_for_test(src)?;
    galaxy.eval_galaxy()
}

pub fn eval_galaxy_src(src: &str) -> Result<Value> {
    let mut galaxy = Galaxy::new(src)?;
    galaxy.eval_galaxy()
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
        let t = Value::Func(Exp::T);
        let f = Value::Func(Exp::F);

        // add
        assert_eq!(eval_src("ap ap add 1 2")?, Value::Num(3));
        assert_eq!(eval_src("ap ap add 3 ap ap add 1 2")?, Value::Num(6));

        // eq
        assert_eq!(eval_src("ap ap eq 1 1")?, t);
        assert_eq!(eval_src("ap ap eq 1 2")?, f);

        // mul
        assert_eq!(eval_src("ap ap mul 2 4")?, Value::Num(8));
        assert_eq!(eval_src("ap ap add 3 ap ap mul 2 4")?, Value::Num(11));

        // div
        assert_eq!(eval_src("ap ap div 4 2")?, Value::Num(2));
        assert_eq!(eval_src("ap ap div 4 3")?, Value::Num(1));
        assert_eq!(eval_src("ap ap div 4 4")?, Value::Num(1));
        assert_eq!(eval_src("ap ap div 4 5")?, Value::Num(0));
        assert_eq!(eval_src("ap ap div 5 2")?, Value::Num(2));
        assert_eq!(eval_src("ap ap div 6 -2")?, Value::Num(-3));
        assert_eq!(eval_src("ap ap div 5 -3")?, Value::Num(-1));
        assert_eq!(eval_src("ap ap div -5 3")?, Value::Num(-1));
        assert_eq!(eval_src("ap ap div -5 -3")?, Value::Num(1));

        // lt
        assert_eq!(eval_src("ap ap lt 0 -1")?, f);
        assert_eq!(eval_src("ap ap lt 0 0")?, f);
        assert_eq!(eval_src("ap ap lt 0 1")?, t);

        Ok(())
    }

    #[test]
    fn eval_unary_test() -> Result<()> {
        // neg
        assert_eq!(eval_src("ap neg 0")?, Value::Num(0));
        assert_eq!(eval_src("ap neg 1")?, Value::Num(-1));
        assert_eq!(eval_src("ap neg -1")?, Value::Num(1));
        assert_eq!(eval_src("ap ap add ap neg 1 2")?, Value::Num(1));

        // inc
        assert_eq!(eval_src("ap inc 0")?, Value::Num(1));
        assert_eq!(eval_src("ap inc 1")?, Value::Num(2));

        // dec
        assert_eq!(eval_src("ap dec 0")?, Value::Num(-1));
        assert_eq!(eval_src("ap dec 1")?, Value::Num(0));

        Ok(())
    }

    #[test]
    fn eval_combinator_test() -> Result<()> {
        // s
        // assert_eq!(eval_src("ap ap ap s add inc 1", EvalResult::Num(3));  // inc is not implemented yet.
        assert_eq!(eval_src("ap ap ap s mul ap add 1 6")?, Value::Num(42));

        // c
        assert_eq!(eval_src("ap ap ap c add 1 2")?, Value::Num(3));

        // b
        // ap ap ap b x0 x1 x2   =   ap x0 ap x1 x2
        // ap ap ap b inc dec x0   =   x0
        assert_eq!(eval_src("ap ap ap b neg neg 1")?, Value::Num(1));

        // t
        // ap ap t x0 x1   =   x0
        // ap ap t 1 5   =   1
        // ap ap t t i   =   t
        // ap ap t t ap inc 5   =   t
        // ap ap t ap inc 5 t   =   6
        assert_eq!(eval_src("ap ap t 1 5")?, Value::Num(1));
        assert_eq!(eval_src("ap ap t t 1")?, Value::Func(Exp::T));
        assert_eq!(eval_src("ap ap t t ap inc 5")?, Value::Func(Exp::T));
        assert_eq!(eval_src("ap ap t ap inc 5 t")?, Value::Num(6));

        // f
        assert_eq!(eval_src("ap ap f 1 2")?, Value::Num(2));

        // i
        assert_eq!(eval_src("ap i 0")?, Value::Num(0));
        assert_eq!(eval_src("ap i i")?, Value::Func(Exp::I));

        Ok(())
    }

    #[test]
    fn eval_cons_test() -> Result<()> {
        // car, cdr, cons
        // car
        // ap car ap ap cons x0 x1   =   x0
        // ap car x2   =   ap x2 t
        assert_eq!(eval_src("ap car ap ap cons 0 1")?, Value::Num(0));
        assert_eq!(eval_src("ap cdr ap ap cons 0 1")?, Value::Num(1));

        // nil
        // ap nil x0   =   t
        assert_eq!(eval_src("ap nil 1")?, Value::Func(Exp::T));

        // isnil
        assert_eq!(eval_src("ap isnil nil")?, Value::Func(Exp::T));
        assert_eq!(eval_src("ap isnil 1")?, Value::Func(Exp::F));

        Ok(())
    }

    #[test]
    fn eval_galaxy_src_test() -> Result<()> {
        let src = ":1 = 2
galaxy = :1
";
        assert_eq!(eval_galaxy_src(src)?, Value::Num(2));

        let src = ":1 = 2
:2 = :1
galaxy = :2
";
        assert_eq!(eval_galaxy_src(src)?, Value::Num(2));

        let src = ":1 = 2
:2 = ap inc :1
galaxy = :2
";
        assert_eq!(eval_galaxy_src(src)?, Value::Num(3));

        let src = ":1 = 2
:2 = ap ap add 1 :1
galaxy = :2
";
        assert_eq!(eval_galaxy_src(src)?, Value::Num(3));

        let src = ":1 = ap add 1
:2 = ap :1 2
galaxy = :2
";
        assert_eq!(eval_galaxy_src(src)?, Value::Num(3));

        Ok(())
    }

    #[test]
    fn eval_recursive_func_test() -> Result<()> {
        // From video part2
        // https://www.youtube.com/watch?v=oU4RAEQCTDE
        let src = ":1 = ap f :1
:2 = ap :1 42
galaxy = :2
";
        assert_eq!(eval_galaxy_src(src)?, Value::Num(42));

        let src = ":1 = ap :1 1
:2 = ap ap t 42 :1
galaxy = :2
";
        assert_eq!(eval_galaxy_src(src)?, Value::Num(42));

        Ok(())
    }

    #[test]
    fn eval_recursive_func_1141_test() -> Result<()> {
        // :1141 = ap ap c b ap ap s ap ap b c ap ap b ap b b ap eq 0 ap ap b ap c :1141 ap add -1

        // Ap(Ap(C, B),
        //    Ap(Ap(S,
        //          Ap(Ap(B, C),
        //             Ap(Ap(B, Ap(B, B)),
        //                Ap(Eq, Num(0))))),
        //       Ap(Ap(B,
        //             Ap(C,
        //                Var(1141))),
        //          Ap(Add, Num(-1)))))

        let src = ":1141 = ap ap c b ap ap s ap ap b c ap ap b ap b b ap eq 0 ap ap b ap c :1141 ap add -1
galaxy = :1141
";
        println!("1141 result: {:?}", eval_galaxy_src(&src));
        Ok(())
    }

    #[test]
    fn run_galaxy_test() -> Result<()> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("task/galaxy.txt");
        let src = std::fs::read_to_string(path)?.trim().to_string();
        println!("galaxy result: {:?}", eval_galaxy_src(&src));
        Ok(())
    }
}
