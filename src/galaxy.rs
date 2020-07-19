use crate::plot;
use crate::prelude::*;
pub use log::*;
use std::convert::From;

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
    Ap(Box<E>, Box<E>),
    Num(i64),
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

use Expr::*;

// Convenient functions
fn ap(e0: impl Into<E>, e1: impl Into<E>) -> Expr {
    Ap(Box::new(e0.into()), Box::new(e1.into()))
}

// #[derive(PartialEq, Clone, Debug)]
#[derive(PartialEq, Clone)]
pub struct E {
    expr: Expr,
    evaluated: bool,
}

impl std::fmt::Debug for E {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.expr)
    }
}

impl From<Expr> for E {
    fn from(expr: Expr) -> Self {
        E {
            expr,
            evaluated: false,
        }
    }
}

fn parse<'a>(tokens: &'a [&'a str]) -> Result<ParseResult<'a>> {
    assert!(!tokens.is_empty());
    let (current_token, tokens) = (tokens[0], &tokens[1..]);
    if current_token == "ap" {
        // TODO: parse overflows in debug build.
        let ParseResult { exp: e0, tokens } = parse(tokens)?;
        let ParseResult { exp: e1, tokens } = parse(tokens)?;
        Ok(ParseResult {
            exp: ap(e0, e1),
            tokens,
        })
    } else {
        Ok(ParseResult {
            exp: match current_token {
                "add" => Add,
                "eq" => Eq,
                "mul" => Mul,
                "div" => Div,
                "lt" => Lt,
                "neg" => Neg,
                "inc" => Inc,
                "dec" => Dec,
                "s" => S,
                "c" => C,
                "b" => B,
                "t" => T,
                "f" => F,
                "i" => I,
                "cons" => Cons,
                "car" => Car,
                "cdr" => Cdr,
                "nil" => Nil,
                "isnil" => Isnil,
                x => {
                    if x.as_bytes()[0] == b':' {
                        let var_id: u64 = x[1..].parse()?;
                        debug!("parsed var_id: {}", var_id);
                        Var(var_id)
                    } else {
                        // TODO: Add context error message.
                        let num: i64 = x.parse().context("number parse error")?;
                        Num(num)
                    }
                }
            },
            tokens,
        })
    }
}

#[derive(PartialEq, Clone, Debug)]
struct InteractResult {
    flag: Expr,
    new_state: Expr,
    images: Expr,
}

impl InteractResult {
    fn new(expr: Expr) -> Result<InteractResult> {
        let destruct = list_destruction(expr)?;
        assert_eq!(destruct.len(), 3);
        let mut iter = destruct.into_iter();
        Ok(InteractResult {
            flag: iter.next().unwrap(),
            new_state: iter.next().unwrap(),
            images: iter.next().unwrap(),
        })
    }
}

pub fn run() -> Result<()> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("task/galaxy.txt");
    let src = std::fs::read_to_string(path)?.trim().to_string();
    let mut galaxy = Galaxy::new(&src)?;
    galaxy.run()
}

struct Galaxy {
    galaxy_id: u64,
    vars: HashMap<u64, E>,
    max_var_id: u64,
}

// TODO: Avoid stackoverflow
/*
fn list_destruction_recursive(expr: Expr) -> Result<Vec<Expr>> {
    fn internal(expr: Expr) -> Result<Vec<Expr>> {
        // it is ap ap cons flag ap ap cons newState ap ap cons data nil

        // List construction
        // https://message-from-space.readthedocs.io/en/latest/message30.html

        // ( x0 , x1 , x2 )   =   ap ap cons x0 ap ap cons x1 ap ap cons x2 nil
        // ( flag , new_state , images )   =   ap ap cons x0 ap ap cons x1 ap ap cons x2 nil

        // https://message-from-space.readthedocs.io/en/latest/message13.html
        match expr {
            Nil => Ok(Vec::new()),
            // 1. [ap ap cons 1 nil]
            // 2. [ap ap cons 1 ap ap cons 2 nil]
            Ap(x0, x1) => match x0.expr {
                // 1. ap [ap cons 1] [nil]
                // 2. ap [ap cons 1 ap ap cons 2] [nil]
                Ap(x0, x2) => match x0.expr {
                    // 1. ap ap [cons] [1] [nil]
                    // 2. ap ap [cons] [1] [ap ap cons 2 nil]
                    Cons => {
                        let mut post = internal(x1.expr)?;
                        post.push(x2.expr);
                        Ok(post)
                    }
                    _ => bail!("can not list destruction"),
                },
                _ => bail!("can not list destruction"),
            },
            _ => bail!("can not list destruction"),
        }
    }

    let mut res = internal(expr)?;
    res.reverse();
    Ok(res)
}
*/

// TODO: Avoid stackoverflow
fn list_destruction(expr: Expr) -> Result<Vec<Expr>> {
    fn internal(expr: Expr) -> Result<Option<(Expr, Expr)>> {
        // it is ap ap cons flag ap ap cons newState ap ap cons data nil

        // List construction
        // https://message-from-space.readthedocs.io/en/latest/message30.html

        // ( x0 , x1 , x2 )   =   ap ap cons x0 ap ap cons x1 ap ap cons x2 nil
        // ( flag , new_state , images )   =   ap ap cons x0 ap ap cons x1 ap ap cons x2 nil

        // https://message-from-space.readthedocs.io/en/latest/message13.html
        match expr {
            Nil => Ok(None),
            // 1. [ap ap cons 1 nil]
            // 2. [ap ap cons 1 ap ap cons 2 nil]
            Ap(x0, x1) => match x0.expr {
                // 1. ap [ap cons 1] [nil]
                // 2. ap [ap cons 1 ap ap cons 2] [nil]
                Ap(x0, x2) => match x0.expr {
                    // 1. ap ap [cons] [1] [nil]
                    // 2. ap ap [cons] [1] [ap ap cons 2 nil]
                    Cons => {
                        Ok(Some((x2.expr, x1.expr)))
                        // let mut post = internal(x1.expr)?;
                        // post.push(x2.expr);
                        // Ok(post)
                    }
                    _ => bail!("can not list destruction"),
                },
                _ => bail!("can not list destruction"),
            },
            _ => bail!("can not list destruction"),
        }
    }

    let mut expr = expr;
    let mut list = Vec::new();
    loop {
        if let Some((x, xs)) = internal(expr)? {
            list.push(x);
            expr = xs;
        } else {
            break;
        }
    }
    Ok(list)
}

// points: [Ap(Ap(Cons, Num(-1)), Num(-3)), Ap(Ap(Cons, Num(0)), Num(-3)), Ap(Ap(Cons, Num(1)), Num(-3)), Ap(Ap(Cons, Num(2)), Num(-2)), Ap(Ap(Cons, Num(-2)), Num(-1)), Ap(Ap(Cons, Num(-1)), Num(-1)), Ap(Ap(Cons, Num(0)), Num(-1)), Ap(Ap(Cons, Num(3)), Num(-1)), Ap(Ap(Cons, Num(-3)), Num(0)), Ap(Ap(Cons, Num(-1)), Num(0)), Ap(Ap(Cons, Num(1)), Num(0)), Ap(Ap(Cons, Num(3)), Num(0)), Ap(Ap(Cons, Num(-3)), Num(1)), Ap(Ap(Cons, Num(0)), Num(1)), Ap(Ap(Cons, Num(1)), Num(1)), Ap(Ap(Cons, Num(2)), Num(1)), Ap(Ap(Cons, Num(-2)), Num(2)), Ap(Ap(Cons, Num(-1)), Num(3)), Ap(Ap(Cons, Num(0)), Num(3)), Ap(Ap(Cons, Num(1)), Num(3)), Ap(Ap(Cons, Num(-7)), Num(-3)), Ap(Ap(Cons, Num(-8)), Num(-2))]

fn images_to_points(images: Expr) -> Result<Vec<(i64, i64)>> {
    let mut points = Vec::new();
    for p in list_destruction(images)? {
        for p in list_destruction(p)? {
            // Ap(Ap(Cons, Num(-1)), Num(-3))
            if let Ap(x, y) = p {
                if let Ap(c, x) = x.expr {
                    if let (Cons, Num(x), Num(y)) = (c.expr, x.expr, y.expr) {
                        points.push((x, y));
                    } else {
                        bail!("can not parse point")
                    }
                } else {
                    bail!("can not parse point")
                }
            } else {
                bail!("can not parse point")
            }
        }
    }
    Ok(points)
}

fn print_images(images: Expr) -> Result<()> {
    let points = images_to_points(images)?;
    info!("points: {:?}", points);
    // plot::plot_galaxy(points)?;
    Ok(())
    // todo!()
}

impl Galaxy {
    fn new_for_test(src: &str) -> Result<Galaxy> {
        Ok(Galaxy {
            galaxy_id: 1,
            vars: {
                let mut vars = HashMap::new();
                vars.insert(1, parse_src(src)?.into());
                vars
            },
            max_var_id: 1,
        })
    }

    fn new(src: &str) -> Result<Galaxy> {
        let lines = src.trim().split('\n').collect::<Vec<_>>();
        // debug!("last line: {}", lines[lines.len() - 1]);
        let galaxy_line_re = Regex::new(r"galaxy *= :*(\d+)$").unwrap();
        let cap = galaxy_line_re.captures(lines[lines.len() - 1]).unwrap();
        let galaxy_id: u64 = cap[1].parse()?;
        // debug!("galaxy_id: {}", galaxy_id);

        let mut max_var_id = 0;

        Ok(Galaxy {
            galaxy_id,
            vars: {
                let mut vars = HashMap::new();
                let re = Regex::new(r":(\d+) *= *(.*)$").unwrap();

                for line in lines.iter().take(lines.len() - 1) {
                    // println!("parse: line: {}", line);
                    let cap = re.captures(line).unwrap();
                    debug!("var: {}", &cap[1]);
                    let var_id = cap[1].parse::<u64>()?;
                    max_var_id = max_var_id.max(var_id);
                    vars.insert(var_id, parse_src(&cap[2])?.into());
                    // println!("{}, {}", &cap[1], &cap[2]);
                }
                vars
            },
            max_var_id,
        })
    }

    fn interact_galaxy(&mut self, state: Expr, event: Expr) -> Result<Expr> {
        let expr = ap(ap(Expr::Var(self.galaxy_id), state), event);
        Ok(self.eval(expr)?.expr)
    }

    fn get_next_event(&self) -> Result<(i64, i64)> {
        loop {
            let mut input = String::new();
            std::io::stdin().read_line(&mut input)?;
            let args = input.trim().split_whitespace().collect::<Vec<_>>();
            if args.len() < 2 {
                error!("invalid args: len < 2");
            }
            if let Ok(x) = args[0].parse::<i64>() {
                if let Ok(y) = args[1].parse::<i64>() {
                    return Ok((x, y));
                }
            }
            error!("invalid line");
        }
    }

    fn run(&mut self) -> Result<()> {
        let mut click_events = vec![
            // 8 times clicks at (0, 0)
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            // click crosshair
            (8, 4),
            (2, -8),
            (3, 6),
            (0, -14),
            (-4, 10),
            // many points 2 <= x <= 17, -7 <= y <= 8
            (9, -3),
            // many points -12 <= x <= 3, 0 <= y <=15
            (-4, 10),
            // many points -2 <= x <= 13, -4 <= y <=11
        ];

        for x in -2..13 {
            for y in -4..11 {
                click_events.push((x, y));
            }
        }
        // for _ in 0..10000 {
        //     events.push((0, 0));
        // }

        // let mut visited = std::collections::HashSet::new();
        // let mut queue = std::collections::VecDeque::new();

        // for event in &events {
        //     queue.push_back(event.clone());
        //     visited.insert(event);
        // }
        let mut state = Nil;

        // let mut got_points = std::collections::HashSet::new();

        let mut points = HashSet::new();

        let mut effective_clicks = Vec::new();

        // while let Some(event) = queue.pop_front() {
        for click in click_events {
            // let event = self.get_next_event()?;
            trace!("click: {:?}", click);
            trace!("state: {:?}", state);
            let (x, y) = click.clone();
            let event = ap(ap(Cons, Num(x)), Num(y));

            let (new_state, images) = self.interact(state.clone(), event)?;

            // print_images(images)?;
            let new_points = images_to_points(images)?
                .into_iter()
                .collect::<HashSet<_>>();
            trace!("new_points: {:?}", new_points);

            if points != new_points || state != new_state {
                effective_clicks.push(click.clone());

                if points != new_points {
                    error!("click {:?} => screen is changed: {:?}", click, new_points);
                }
                if state != new_state {
                    warn!("click {:?} => new state: {:?}", click, new_state);
                }
                // plot::plot_galaxy(new_points.iter().cloned().collect())?;
            }

            points = new_points;
            state = new_state;
        }

        // info!("points: {:?}", got_points);
        // plot::plot_galaxy(got_points.into_iter().collect())?;

        error!("effective_clicks: {:?}", effective_clicks);
        plot::plot_galaxy(points.iter().cloned().collect())?;
        Ok(())
    }

    fn interact(&mut self, state: Expr, event: Expr) -> Result<(Expr, Expr)> {
        debug!("> interact");
        let expr = self.interact_galaxy(state, event)?;
        let interact_result = InteractResult::new(expr)?;
        if interact_result.flag == Num(0) {
            debug!("< interact");
            Ok((interact_result.new_state, interact_result.images))
        } else {
            error!("flag is not zero: {:?}", interact_result.flag);
            // https://message-from-space.readthedocs.io/en/latest/implementation.html
            // Send_TO_ALIEN_PROXY
            todo!()
        }
    }

    fn eval_galaxy(&mut self) -> Result<E> {
        self.eval_var(self.galaxy_id)
    }

    fn eval_var(&mut self, id: u64) -> Result<E> {
        trace!("eval var: {}", id);
        let entry = self.vars[&id].clone();
        if entry.evaluated {
            trace!("eval var: {} -> evaluated: {:?}", id, entry.expr);
            Ok(entry)
        } else {
            trace!("eval var: {} -> toeval: {:?}", id, entry.expr);
            let res = self.eval(entry.expr)?;
            trace!(
                "eval var: {} <- evaluated? {} {:?}",
                id,
                res.evaluated,
                res.expr
            );
            assert!(self.vars.insert(id, res.clone()).is_some());
            Ok(res)
        }
    }

    fn add_new_var(&mut self, expr: Expr) -> u64 {
        self.max_var_id += 1;
        let new_id = self.max_var_id;
        assert!(self.vars.insert(new_id, expr.into()).is_none());
        new_id
    }

    fn eval(&mut self, e: impl Into<E>) -> Result<E> {
        let e: E = e.into();
        if e.evaluated {
            trace!("eval: {:?} -> evaluated", e);
            Ok(e)
        } else {
            trace!("eval: {:?} -> toeval", e);
            let res = self.eval_internal(e.expr)?;
            trace!("eval result: {:?}", res);
            let mut res: E = res.into();
            res.evaluated = true;
            Ok(res)
        }
    }

    fn eval_internal(&mut self, e: Expr) -> Result<Expr> {
        match e {
            Ap(left, right) => self.apply(*left, *right),
            Var(n) => Ok(self.eval_var(n)?.expr),
            x => Ok(x),
        }
    }

    fn apply(&mut self, f: impl Into<E>, x0: impl Into<E>) -> Result<Expr> {
        let f: E = f.into();
        let x0: E = x0.into();
        trace!("apply: f: {:?}, x0: {:?}", f, x0);
        let f = self.eval(f)?;
        match f.expr {
            Num(_) => bail!("can not apply: nuber"),
            Neg => match self.eval(x0)?.expr {
                Num(n) => Ok(Num(-n)),
                _ => bail!("can not apply: neg"),
            },
            Inc => match self.eval(x0)?.expr {
                Num(n) => Ok(Num(n + 1)),
                _ => bail!("can not apply: inc"),
            },
            Dec => match self.eval(x0)?.expr {
                Num(n) => Ok(Num(n - 1)),
                _ => bail!("can not apply: dec"),
            },
            I => Ok(self.eval(x0)?.expr),
            // ap car x2 = ap x2 t
            Car => self.apply(x0, T),
            Cdr => self.apply(x0, F),
            Nil => Ok(T),
            Isnil => match self.eval(x0)?.expr {
                Nil => Ok(T),
                _ => Ok(F),
            },
            Ap(exp, e0) => {
                let (e0, e1): (E, E) = (*e0, x0); // For readability.
                let f = self.eval(*exp)?;
                match f.expr {
                    Add => match (self.eval(e0)?.expr, self.eval(e1)?.expr) {
                        (Num(n0), Num(n1)) => Ok(Num(n0 + n1)),
                        _ => bail!("can not apply: add"),
                    },
                    Mul => match (self.eval(e0)?.expr, self.eval(e1)?.expr) {
                        (Num(n0), Num(n1)) => Ok(Num(n0 * n1)),
                        // _ => bail!("can not apply: mul"),
                        (x, y) => bail!("can not apply: mul: e0: {:?}, e1: {:?}", x, y),
                    },
                    Div => match (self.eval(e0)?.expr, self.eval(e1)?.expr) {
                        (Num(n0), Num(n1)) => Ok(Num(n0 / n1)),
                        _ => bail!("can not apply: div"),
                    },
                    Eq => match (self.eval(e0)?.expr, self.eval(e1)?.expr) {
                        (Num(n0), Num(n1)) => {
                            if n0 == n1 {
                                Ok(T)
                            } else {
                                Ok(F)
                            }
                        }
                        _ => bail!("can not apply: eq"),
                    },
                    Lt => match (self.eval(e0)?.expr, self.eval(e1)?.expr) {
                        (Num(n0), Num(n1)) => {
                            if n0 < n1 {
                                Ok(T)
                            } else {
                                Ok(F)
                            }
                        }
                        _ => bail!("can not apply: lt"),
                    },
                    T => Ok(self.eval(e0)?.expr),
                    F => Ok(self.eval(e1)?.expr),
                    S => Ok(ap(ap(S, e0), e1)),
                    C => Ok(ap(ap(C, e0), e1)),
                    B => Ok(ap(ap(B, e0), e1)),
                    // Cons => Ok(ap(ap(Cons, e0), e1)),
                    // Eval cons earger
                    Cons => Ok(ap(ap(Cons, self.eval(e0)?), self.eval(e1)?)),
                    Ap(exp, e) => {
                        let (e0, e1, e2): (E, E, E) = (*e, e0, e1);
                        let f = self.eval(*exp)?;
                        match f.expr {
                            S => {
                                trace!("expand s-combinator: {:?}", (&e0, &e1, &e2));
                                // ap ap ap s x0 x1 x2   =   ap ap x0 x2 ap x1 x2
                                // ap ap ap s add inc 1   =   3
                                let e2: E = {
                                    if e2.evaluated {
                                        trace!("s-combinator: e2 is already evaluated)");
                                        e2
                                    } else {
                                        trace!("s-combinator: e2 is not evaluated");
                                        Var(self.add_new_var(e2.expr)).into()
                                    }
                                };
                                trace!("s-combin!ator: E2: {:?}", e2);

                                let ap_x0_x2 = ap(e0, e2.clone());
                                let ap_x1_x2 = ap(e1, e2);
                                self.apply(ap_x0_x2, ap_x1_x2)
                            }
                            C => {
                                // ap ap ap c x0 x1 x2   =   ap ap x0 x2 x1
                                // ap ap ap c add 1 2   =   3
                                self.apply(ap(e0, e2), e1)
                            }
                            B => {
                                // ap ap ap b x0 x1 x2   =   ap x0 ap x1 x2
                                // ap ap ap b inc dec x0   =   x0
                                self.apply(e0, ap(e1, e2))
                            }
                            Cons => {
                                // cons
                                // ap ap ap cons x0 x1 x2   =   ap ap x2 x0 x1
                                self.apply(ap(e2, e0), e1)
                            }
                            _ => bail!("can not apply: ap ap ap"),
                        }
                    }
                    _ => bail!("can not apply: ap ap"),
                }
            }
            f => Ok(ap(f, x0)),
        }
    }
}

pub fn eval_src(src: &str) -> Result<Expr> {
    // let exp = parse_src(src)?;
    // eval(exp)
    let mut galaxy = Galaxy::new_for_test(src)?;
    galaxy.eval_galaxy().map(|e| e.expr)
}

pub fn eval_galaxy_src(src: &str) -> Result<Expr> {
    let mut galaxy = Galaxy::new(src)?;
    galaxy.eval_galaxy().map(|e| e.expr)
}

// fn send(s: &str) -> String {
//     todo!()
// }

#[cfg(test)]
mod tests {

    use super::*;
    use chrono::Local;
    use std::io::Write as _;

    #[allow(dead_code)]
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
        assert_eq!(parse_src("1")?, Num(1));
        assert_eq!(parse_src("add")?, Add);
        assert_eq!(parse_src("ap ap add 1 2")?, ap(ap(Add, Num(1)), Num(2)));
        assert_eq!(parse_src("ap ap eq 1 2")?, ap(ap(Eq, Num(1)), Num(2)));
        assert!(parse_src("add 1").is_err());
        Ok(())
    }

    #[test]
    fn eval_test() -> Result<()> {
        // add
        assert_eq!(eval_src("ap ap add 1 2")?, Num(3));
        assert_eq!(eval_src("ap ap add 3 ap ap add 1 2")?, Num(6));

        // eq
        assert_eq!(eval_src("ap ap eq 1 1")?, T);
        assert_eq!(eval_src("ap ap eq 1 2")?, F);

        // mul
        assert_eq!(eval_src("ap ap mul 2 4")?, Num(8));
        assert_eq!(eval_src("ap ap add 3 ap ap mul 2 4")?, Num(11));

        // div
        assert_eq!(eval_src("ap ap div 4 2")?, Num(2));
        assert_eq!(eval_src("ap ap div 4 3")?, Num(1));
        assert_eq!(eval_src("ap ap div 4 4")?, Num(1));
        assert_eq!(eval_src("ap ap div 4 5")?, Num(0));
        assert_eq!(eval_src("ap ap div 5 2")?, Num(2));
        assert_eq!(eval_src("ap ap div 6 -2")?, Num(-3));
        assert_eq!(eval_src("ap ap div 5 -3")?, Num(-1));
        assert_eq!(eval_src("ap ap div -5 3")?, Num(-1));
        assert_eq!(eval_src("ap ap div -5 -3")?, Num(1));

        // lt
        assert_eq!(eval_src("ap ap lt 0 -1")?, F);
        assert_eq!(eval_src("ap ap lt 0 0")?, F);
        assert_eq!(eval_src("ap ap lt 0 1")?, T);

        Ok(())
    }

    #[test]
    fn eval_unary_test() -> Result<()> {
        // neg
        assert_eq!(eval_src("ap neg 0")?, Num(0));
        assert_eq!(eval_src("ap neg 1")?, Num(-1));
        assert_eq!(eval_src("ap neg -1")?, Num(1));
        assert_eq!(eval_src("ap ap add ap neg 1 2")?, Num(1));

        // inc
        assert_eq!(eval_src("ap inc 0")?, Num(1));
        assert_eq!(eval_src("ap inc 1")?, Num(2));

        // dec
        assert_eq!(eval_src("ap dec 0")?, Num(-1));
        assert_eq!(eval_src("ap dec 1")?, Num(0));

        Ok(())
    }

    #[test]
    fn eval_s_c_b_test() -> Result<()> {
        // s
        assert_eq!(eval_src("ap ap ap s add inc 1")?, Num(3));
        assert_eq!(eval_src("ap ap ap s mul ap add 1 6")?, Num(42));

        // c
        assert_eq!(eval_src("ap ap ap c add 1 2")?, Num(3));

        // b
        // ap ap ap b x0 x1 x2   =   ap x0 ap x1 x2
        // ap ap ap b inc dec x0   =   x0
        assert_eq!(eval_src("ap ap ap b neg neg 1")?, Num(1));
        Ok(())
    }

    #[test]
    fn eval_t_f_i_test() -> Result<()> {
        // t
        // ap ap t x0 x1   =   x0
        // ap ap t 1 5   =   1
        // ap ap t t i   =   t
        // ap ap t t ap inc 5   =   t
        // ap ap t ap inc 5 t   =   6
        assert_eq!(eval_src("ap ap t 1 5")?, Num(1));
        assert_eq!(eval_src("ap ap t t 1")?, T);
        assert_eq!(eval_src("ap ap t t ap inc 5")?, T);
        assert_eq!(eval_src("ap ap t ap inc 5 t")?, Num(6));

        // f
        assert_eq!(eval_src("ap ap f 1 2")?, Num(2));

        // i
        assert_eq!(eval_src("ap i 0")?, Num(0));
        assert_eq!(eval_src("ap i i")?, I);

        Ok(())
    }

    #[test]
    fn eval_cons_test() -> Result<()> {
        // car, cdr, cons
        // car
        // ap car ap ap cons x0 x1   =   x0
        // ap car x2   =   ap x2 t
        assert_eq!(eval_src("ap car ap ap cons 0 1")?, Num(0));
        assert_eq!(eval_src("ap cdr ap ap cons 0 1")?, Num(1));

        // nil
        // ap nil x0   =   t
        assert_eq!(eval_src("ap nil 1")?, T);

        // isnil
        assert_eq!(eval_src("ap isnil nil")?, T);
        assert_eq!(eval_src("ap isnil 1")?, F);

        Ok(())
    }

    #[test]
    fn eval_galaxy_src_test() -> Result<()> {
        let src = ":1 = 2
    galaxy = :1
    ";
        assert_eq!(eval_galaxy_src(src)?, Num(2));

        let src = ":1 = 2
    :2 = :1
    galaxy = :2
    ";
        assert_eq!(eval_galaxy_src(src)?, Num(2));

        let src = ":1 = 2
    :2 = ap inc :1
    galaxy = :2
    ";
        assert_eq!(eval_galaxy_src(src)?, Num(3));

        let src = ":1 = 2
    :2 = ap ap add 1 :1
    galaxy = :2
    ";
        assert_eq!(eval_galaxy_src(src)?, Num(3));

        let src = ":1 = ap add 1
    :2 = ap :1 2
    galaxy = :2
    ";
        assert_eq!(eval_galaxy_src(src)?, Num(3));

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
        assert_eq!(eval_galaxy_src(src)?, Num(42));

        let src = ":1 = ap :1 1
    :2 = ap ap t 42 :1
    galaxy = :2
    ";
        assert_eq!(eval_galaxy_src(src)?, Num(42));

        Ok(())
    }

    #[test]
    fn eval_recursive_func_1141_test() -> Result<()> {
        // init_env_logger();

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
        let result = eval_galaxy_src(&src)?;
        // println!("1141 result: {:?}", result);
        assert_eq!(
                format!("{:?}", result),
                "Ap(Ap(C, B), Ap(Ap(S, Ap(Ap(B, C), Ap(Ap(B, Ap(B, B)), Ap(Eq, Num(0))))), Ap(Ap(B, Ap(C, Var(1141))), Ap(Add, Num(-1)))))");
        Ok(())
    }

    #[test]
    fn eval_galaxy_test() -> Result<()> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("task/galaxy.txt");
        let src = std::fs::read_to_string(path)?.trim().to_string();

        let result = eval_galaxy_src(&src)?;
        // println!("galaxy result: {:?}", result);
        assert_eq!(
            format!("{:?}", result),
            "Ap(Ap(C, Ap(Ap(B, C), Ap(Ap(C, Ap(Ap(B, C), Var(1342))), Var(1328)))), Var(1336))"
        );
        Ok(())
    }

    #[test]
    fn interact_galaxy_test() -> Result<()> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("task/galaxy.txt");
        let src = std::fs::read_to_string(path)?.trim().to_string();

        let mut galaxy = Galaxy::new(&src)?;
        let res = galaxy.interact_galaxy(Nil, ap(ap(Cons, Num(0)), Num(0)))?;
        // galaxy interact result: PartialAp2(Cons, Num(0), Ap(Ap(Ap(Ap(C, Ap(Ap(B, B), Cons)), Ap(Ap(C, Cons), Nil)), Ap(Ap(Ap(Ap(C, Ap(Ap(B, B), Ap(Ap(C, Var(1144)), Num(1)))), Ap(Ap(C, Cons), Nil)), Var(410)), Var(429))), Ap(Var(1229), Var(429))))

        // it is ap ap cons flag ap ap cons newState ap ap cons data nil

        assert_eq!(
            format!("{:?}", res),
            "Ap(Ap(Cons, Num(0)), Ap(Ap(Cons, Ap(Ap(Cons, Num(0)), Ap(Ap(Cons, Ap(Ap(Cons, Num(0)), Nil)), Ap(Ap(Cons, Num(0)), Ap(Ap(Cons, Nil), Nil))))), Ap(Ap(Cons, Ap(Ap(Cons, Ap(Ap(Cons, Ap(Ap(Cons, Num(-1)), Num(-3))), Ap(Ap(Cons, Ap(Ap(Cons, Num(0)), Num(-3))), Ap(Ap(Cons, Ap(Ap(Cons, Num(1)), Num(-3))), Ap(Ap(Cons, Ap(Ap(Cons, Num(2)), Num(-2))), Ap(Ap(Cons, Ap(Ap(Cons, Num(-2)), Num(-1))), Ap(Ap(Cons, Ap(Ap(Cons, Num(-1)), Num(-1))), Ap(Ap(Cons, Ap(Ap(Cons, Num(0)), Num(-1))), Ap(Ap(Cons, Ap(Ap(Cons, Num(3)), Num(-1))), Ap(Ap(Cons, Ap(Ap(Cons, Num(-3)), Num(0))), Ap(Ap(Cons, Ap(Ap(Cons, Num(-1)), Num(0))), Ap(Ap(Cons, Ap(Ap(Cons, Num(1)), Num(0))), Ap(Ap(Cons, Ap(Ap(Cons, Num(3)), Num(0))), Ap(Ap(Cons, Ap(Ap(Cons, Num(-3)), Num(1))), Ap(Ap(Cons, Ap(Ap(Cons, Num(0)), Num(1))), Ap(Ap(Cons, Ap(Ap(Cons, Num(1)), Num(1))), Ap(Ap(Cons, Ap(Ap(Cons, Num(2)), Num(1))), Ap(Ap(Cons, Ap(Ap(Cons, Num(-2)), Num(2))), Ap(Ap(Cons, Ap(Ap(Cons, Num(-1)), Num(3))), Ap(Ap(Cons, Ap(Ap(Cons, Num(0)), Num(3))), Ap(Ap(Cons, Ap(Ap(Cons, Num(1)), Num(3))), Nil))))))))))))))))))))), Ap(Ap(Cons, Ap(Ap(Cons, Ap(Ap(Cons, Num(-7)), Num(-3))), Ap(Ap(Cons, Ap(Ap(Cons, Num(-8)), Num(-2))), Nil))), Ap(Ap(Cons, Nil), Nil)))), Nil)))"
        );

        println!("galaxy interact result: {:?}", res);

        let interact_result = InteractResult::new(res)?;
        assert_eq!(interact_result.flag, Num(0));
        // assert_eq!(list_destruction(interact_result.new_state)?, vec![Num(0)]);
        Ok(())
    }

    #[test]
    #[ignore]
    fn run_galaxy_test() -> Result<()> {
        run()
    }

    #[test]
    fn list_destruction_test() -> Result<()> {
        assert_eq!(list_destruction(Nil)?, Vec::new());
        assert_eq!(
            list_destruction(parse_src("ap ap cons 1 nil")?)?,
            vec![Num(1)]
        );
        assert_eq!(
            list_destruction(parse_src("ap ap cons 1 ap ap cons 2 nil")?)?,
            vec![Num(1), Num(2)]
        );
        assert_eq!(
            list_destruction(parse_src("ap ap cons 1 ap ap cons 2 ap ap cons 3 nil")?)?,
            vec![Num(1), Num(2), Num(3)]
        );

        Ok(())
    }
}
