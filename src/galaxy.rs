use crate::prelude::*;
pub use log::*;

use regex::Regex;

fn tokenize(src: &str) -> Vec<&str> {
    src.split_whitespace().collect()
}

fn parse_src(src: &str) -> Result<Expr> {
    let tokens = tokenize(src);
    let parse_result = parse(&tokens)?;
    ensure!(
        parse_result.tokens.is_empty(),
        format!("tokens are not consumed: {:?}", parse_result.tokens)
    );
    Ok(parse_result.exp)
}

struct ParseResult<'a> {
    exp: Expr,
    tokens: &'a [&'a str],
}

#[derive(PartialEq, Clone, Debug)]
pub enum Expr {
    Num(i64),
    Ap(Box<Expr>, Box<Expr>),
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
    Func(Expr),
    PartialAp1(Expr, Expr),
    PartialAp2(Expr, Expr, Expr),
}

fn parse<'a>(tokens: &'a [&'a str]) -> Result<ParseResult<'a>> {
    assert!(!tokens.is_empty());
    let (current_token, tokens) = (tokens[0], &tokens[1..]);
    if current_token == "ap" {
        // TODO: parse overflows in debug build.
        let ParseResult { exp: exp1, tokens } = parse(tokens)?;
        let ParseResult { exp: exp2, tokens } = parse(tokens)?;
        Ok(ParseResult {
            exp: Expr::Ap(Box::new(exp1), Box::new(exp2)),
            tokens,
        })
    } else {
        Ok(ParseResult {
            exp: match current_token {
                "add" => Expr::Add,
                "eq" => Expr::Eq,
                "mul" => Expr::Mul,
                "div" => Expr::Div,
                "lt" => Expr::Lt,
                "neg" => Expr::Neg,
                "inc" => Expr::Inc,
                "dec" => Expr::Dec,
                "s" => Expr::S,
                "c" => Expr::C,
                "b" => Expr::B,
                "t" => Expr::T,
                "f" => Expr::F,
                "i" => Expr::I,
                "cons" => Expr::Cons,
                "car" => Expr::Car,
                "cdr" => Expr::Cdr,
                "nil" => Expr::Nil,
                "isnil" => Expr::Isnil,
                x => {
                    if x.as_bytes()[0] == b':' {
                        let var_id: u64 = x[1..].parse()?;
                        Expr::Var(var_id)
                    } else {
                        // TODO: Add context error message.
                        let num: i64 = x.parse().context("number parse error")?;
                        Expr::Num(num)
                    }
                }
            },
            tokens,
        })
    }
}

struct Galaxy {
    galaxy_id: u64,
    functions: HashMap<u64, Expr>,
    results: HashMap<u64, Value>,
}

impl Galaxy {
    fn new_for_test(src: &str) -> Result<Galaxy> {
        let exp = parse_src(src)?;
        Ok(Galaxy {
            galaxy_id: 1,
            functions: {
                let mut map = HashMap::new();
                map.insert(1, exp);
                map
            },
            results: HashMap::new(),
        })
    }

    fn eval_galaxy(&mut self) -> Result<Value> {
        self.eval(self.functions[&self.galaxy_id].clone())
    }

    fn new(src: &str) -> Result<Galaxy> {
        let lines = src.trim().split('\n').collect::<Vec<_>>();
        // println!("last line: {}", lines[lines.len() - 1]);
        let galaxy_line_re = Regex::new(r"galaxy *= :*(\d+)$").unwrap();
        let cap = galaxy_line_re.captures(lines[lines.len() - 1]).unwrap();
        let galaxy_id: u64 = cap[1].parse()?;
        // println!("galaxy_id: {}", galaxy_id);

        Ok(Galaxy {
            galaxy_id,
            functions: {
                let mut map = HashMap::new();
                let re = Regex::new(r":(\d+) *= *(.*)$").unwrap();

                for line in lines.iter().take(lines.len() - 1) {
                    // println!("parse: line: {}", line);
                    let cap = re.captures(line).unwrap();
                    map.insert(cap[1].parse::<u64>()?, parse_src(&cap[2])?);
                    // println!("{}, {}", &cap[1], &cap[2]);
                }
                map
            },
            results: HashMap::new(),
        })
    }

    fn eval(&mut self, exp: Expr) -> Result<Value> {
        // println!("eval: {:?}", exp);
        match exp {
            Expr::Num(n) => Ok(Value::Num(n)),
            Expr::Ap(left, right) => self.apply(*left, *right),
            Expr::Add => Ok(Value::Func(Expr::Add)),
            Expr::Mul => Ok(Value::Func(Expr::Mul)),
            Expr::Div => Ok(Value::Func(Expr::Div)),
            Expr::Eq => Ok(Value::Func(Expr::Eq)),
            Expr::Lt => Ok(Value::Func(Expr::Lt)),
            Expr::Neg => Ok(Value::Func(Expr::Neg)),
            Expr::Inc => Ok(Value::Func(Expr::Inc)),
            Expr::Dec => Ok(Value::Func(Expr::Dec)),
            Expr::S => Ok(Value::Func(Expr::S)),
            Expr::C => Ok(Value::Func(Expr::C)),
            Expr::B => Ok(Value::Func(Expr::B)),
            Expr::T => Ok(Value::Func(Expr::T)),
            Expr::F => Ok(Value::Func(Expr::F)),
            Expr::I => Ok(Value::Func(Expr::I)),
            Expr::Cons => Ok(Value::Func(Expr::Cons)),
            Expr::Car => Ok(Value::Func(Expr::Car)),
            Expr::Cdr => Ok(Value::Func(Expr::Cdr)),
            Expr::Nil => Ok(Value::Func(Expr::Nil)),
            Expr::Isnil => Ok(Value::Func(Expr::Isnil)),
            Expr::Var(n) => self.eval_var(n),
        }
    }

    fn eval_var(&mut self, variable_id: u64) -> Result<Value> {
        if let Some(result) = self.results.get(&variable_id) {
            Ok(result.clone())
        } else {
            let result = self.eval(self.functions[&variable_id].clone())?;
            self.results.insert(variable_id, result.clone());
            Ok(result)
        }
    }

    fn apply(&mut self, f: Expr, x0: Expr) -> Result<Value> {
        debug!("apply: f: {:?}, x0: {:?}", f, x0);
        let f = self.eval(f)?;
        match f {
            Value::Func(exp) => match exp {
                Expr::Neg => match self.eval(x0)? {
                    Value::Num(n) => Ok(Value::Num(-n)),
                    _ => bail!("can not apply"),
                },
                Expr::Inc => match self.eval(x0)? {
                    Value::Num(n) => Ok(Value::Num(n + 1)),
                    _ => bail!("can not apply"),
                },
                Expr::Dec => match self.eval(x0)? {
                    Value::Num(n) => Ok(Value::Num(n - 1)),
                    _ => bail!("can not apply"),
                },
                Expr::I => self.eval(x0),
                // ap car x2 = ap x2 t
                Expr::Car => self.apply(x0, Expr::T),
                Expr::Cdr => self.apply(x0, Expr::F),
                Expr::Nil => Ok(Value::Func(Expr::T)),
                Expr::Isnil => match self.eval(x0)? {
                    Value::Func(Expr::Nil) => Ok(Value::Func(Expr::T)),
                    _ => Ok(Value::Func(Expr::F)),
                },
                exp => Ok(Value::PartialAp1(exp, x0)),
            },
            Value::PartialAp1(exp, e0) => {
                let e1 = x0; // For readability.
                match exp {
                    Expr::Add => match (self.eval(e0)?, self.eval(e1)?) {
                        (Value::Num(n0), Value::Num(n1)) => Ok(Value::Num(n0 + n1)),
                        _ => bail!("can not apply"),
                    },
                    Expr::Mul => match (self.eval(e0)?, self.eval(e1)?) {
                        (Value::Num(n0), Value::Num(n1)) => Ok(Value::Num(n0 * n1)),
                        _ => bail!("can not apply"),
                    },
                    Expr::Div => match (self.eval(e0)?, self.eval(e1)?) {
                        (Value::Num(n0), Value::Num(n1)) => Ok(Value::Num(n0 / n1)),
                        _ => bail!("can not apply"),
                    },
                    Expr::Eq => match (self.eval(e0)?, self.eval(e1)?) {
                        (Value::Num(n0), Value::Num(n1)) => {
                            if n0 == n1 {
                                Ok(Value::Func(Expr::T))
                            } else {
                                Ok(Value::Func(Expr::F))
                            }
                        }
                        _ => bail!("can not apply"),
                    },
                    Expr::Lt => match (self.eval(e0)?, self.eval(e1)?) {
                        (Value::Num(n0), Value::Num(n1)) => {
                            if n0 < n1 {
                                Ok(Value::Func(Expr::T))
                            } else {
                                Ok(Value::Func(Expr::F))
                            }
                        }
                        _ => bail!("can not apply"),
                    },
                    Expr::S => Ok(Value::PartialAp2(Expr::S, e0, e1)),
                    Expr::C => Ok(Value::PartialAp2(Expr::C, e0, e1)),
                    Expr::B => Ok(Value::PartialAp2(Expr::B, e0, e1)),
                    Expr::T => self.eval(e0),
                    // New
                    Expr::F => self.eval(e1),
                    Expr::Cons => Ok(Value::PartialAp2(Expr::Cons, e0, e1)),
                    exp => bail!("can not apply: exp: {:?}, e0: {:?}, e1: {:?}", exp, e0, e1),
                }
            }
            Value::PartialAp2(exp, e0, e1) => {
                let e2 = x0;
                match exp {
                    Expr::S => {
                        // ap ap ap s x0 x1 x2   =   ap ap x0 x2 ap x1 x2
                        // ap ap ap s add inc 1   =   3

                        // TODO: Avoid clone. Use Rc?
                        let ap_x0_x2 = Expr::Ap(Box::new(e0), Box::new(e2.clone()));
                        let ap_x1_x2 = Expr::Ap(Box::new(e1), Box::new(e2));
                        self.apply(ap_x0_x2, ap_x1_x2)
                    }
                    Expr::C => {
                        // ap ap ap c x0 x1 x2   =   ap ap x0 x2 x1
                        // ap ap ap c add 1 2   =   3
                        self.apply(Expr::Ap(Box::new(e0), Box::new(e2)), e1)
                    }
                    Expr::B => {
                        // ap ap ap b x0 x1 x2   =   ap x0 ap x1 x2
                        // ap ap ap b inc dec x0   =   x0
                        self.apply(e0, Expr::Ap(Box::new(e1), Box::new(e2)))
                    }
                    Expr::Cons => {
                        // cons
                        // ap ap ap cons x0 x1 x2   =   ap ap x2 x0 x1
                        self.apply(Expr::Ap(Box::new(e2), Box::new(e0)), e1)
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
    use chrono::Local;
    use std::io::Write as _;

    fn init_env_logger() {
        let _ = env_logger::builder()
            .format(|buf, record| {
                writeln!(
                    buf,
                    "[{} {:5} {}] ({}:{}) {}",
                    Local::now().format("%+"),
                    // record.level(),
                    buf.default_styled_level(record.level()),
                    record.target(),
                    record.file().unwrap_or("unknown"),
                    record.line().unwrap_or(0),
                    record.args(),
                )
            })
            .is_test(true)
            .try_init();
    }

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
        assert_eq!(parse_src("1")?, Expr::Num(1));
        assert_eq!(parse_src("add")?, Expr::Add);
        assert_eq!(
            parse_src("ap ap add 1 2")?,
            Expr::Ap(
                Box::new(Expr::Ap(Box::new(Expr::Add), Box::new(Expr::Num(1)))),
                Box::new(Expr::Num(2))
            )
        );
        assert_eq!(
            parse_src("ap ap eq 1 2")?,
            Expr::Ap(
                Box::new(Expr::Ap(Box::new(Expr::Eq), Box::new(Expr::Num(1)))),
                Box::new(Expr::Num(2))
            )
        );
        assert!(parse_src("add 1").is_err());
        Ok(())
    }

    #[test]
    fn eval_test() -> Result<()> {
        let t = Value::Func(Expr::T);
        let f = Value::Func(Expr::F);

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
        assert_eq!(eval_src("ap ap t t 1")?, Value::Func(Expr::T));
        assert_eq!(eval_src("ap ap t t ap inc 5")?, Value::Func(Expr::T));
        assert_eq!(eval_src("ap ap t ap inc 5 t")?, Value::Num(6));

        // f
        assert_eq!(eval_src("ap ap f 1 2")?, Value::Num(2));

        // i
        assert_eq!(eval_src("ap i 0")?, Value::Num(0));
        assert_eq!(eval_src("ap i i")?, Value::Func(Expr::I));

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
        assert_eq!(eval_src("ap nil 1")?, Value::Func(Expr::T));

        // isnil
        assert_eq!(eval_src("ap isnil nil")?, Value::Func(Expr::T));
        assert_eq!(eval_src("ap isnil 1")?, Value::Func(Expr::F));

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
        init_env_logger();

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
        // 1141 result: Ok(PartialAp2(C, B, Ap(Ap(S, Ap(Ap(B, C), Ap(Ap(B, Ap(B, B)), Ap(Eq, Num(0))))), Ap(Ap(B, Ap(C, Var(1141))), Ap(Add, Num(-1))))))
        Ok(())
    }

    #[test]
    fn run_galaxy_test() -> Result<()> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("task/galaxy.txt");
        let src = std::fs::read_to_string(path)?.trim().to_string();
        println!("galaxy result: {:?}", eval_galaxy_src(&src));
        // [2020-07-19 Sun] galaxy result: Ok(PartialAp2(C, Ap(Ap(B, C), Ap(Ap(C, Ap(Ap(B, C), Var(1342))), Var(1328))), Var(1336)))
        Ok(())
    }
}
