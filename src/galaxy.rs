use crate::api;
use crate::prelude::*;
use log::*;
use regex::Regex;
use std::convert::From;
use std::rc::Rc;

#[derive(PartialEq, Clone, Debug)]
pub enum Expr {
    Ap(Rc<Expr>, Rc<Expr>, bool),
    Var(usize),
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
}

use Expr::*;

// Convenient functions
fn ap(e0: Rc<Expr>, e1: Rc<Expr>) -> Rc<Expr> {
    Rc::new(Ap(e0, e1, false))
}

fn tokenize(src: &str) -> Vec<&str> {
    src.split_whitespace().collect()
}

fn parse_src(src: &str) -> Result<Rc<Expr>> {
    let tokens = tokenize(src);
    let ParseResult { exp, tokens } = parse(&tokens)?;
    ensure!(
        tokens.is_empty(),
        format!("tokens are not consumed: {:?}", tokens)
    );
    Ok(exp)
}

struct ParseResult<'a> {
    exp: Rc<Expr>,
    tokens: &'a [&'a str],
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
            exp: Rc::new(match current_token {
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
                        let var_id: usize = x[1..].parse()?;
                        debug!("parsed var_id: {}", var_id);
                        Var(var_id)
                    } else {
                        // TODO: Add context error message.
                        let num: i64 = x.parse().context("number parse error")?;
                        Num(num)
                    }
                }
            }),
            tokens,
        })
    }
}

impl Expr {
    fn destruct_cons(self: &Rc<Self>) -> Result<Option<(Rc<Expr>, Rc<Expr>)>> {
        match self.as_ref() {
            Nil => Ok(None),
            Ap(a, cdr, _) => match a.as_ref() {
                Ap(cons, car, _) if cons.as_ref() == &Cons => Ok(Some((car.clone(), cdr.clone()))),
                _ => bail!("invalid structure"),
            },
            _ => bail!("invalid structure"),
        }
    }

    fn to_vec(self: &Rc<Self>) -> Result<Vec<Rc<Expr>>> {
        let mut expr = self.clone();
        let mut list = Vec::new();
        while let Some((x, xs)) = expr.destruct_cons()? {
            list.push(x);
            expr = xs;
        }
        Ok(list)
    }

    fn to_screen(self: &Rc<Self>) -> Result<Screen> {
        let mut screen = Screen::new();
        for image_expr in self.to_vec()? {
            let mut image = Image::new();
            for p in image_expr.to_vec()? {
                if let Some((x, y)) = p.destruct_cons()? {
                    if let (Num(x), Num(y)) = (x.as_ref(), y.as_ref()) {
                        image.push((*x, *y));
                    } else {
                        bail!("invalid point structure")
                    }
                } else {
                    bail!("invalid point structure")
                }
            }
            screen.push(image)
        }
        Ok(screen)
    }

    fn size(&self) -> u64 {
        match self {
            Ap(a, b, _) => 1 + a.size() + b.size(),
            _ => 1,
        }
    }
}

trait Demodulatable {
    fn demodulate(self) -> Result<Rc<Expr>>;
}

impl Demodulatable for &str {
    // 11: ap ap cons
    // 00: nil
    fn demodulate(self) -> Result<Rc<Expr>> {
        let s: &str = self;
        let mut pos = 0;
        let mut tokens = String::new();
        while pos < s.len() {
            match &s[pos..(pos + 2)] {
                "00" => {
                    tokens.push_str(" nil");
                    pos += 2;
                }
                "11" => {
                    tokens.push_str(" ap ap cons");
                    pos += 2;
                }
                sign => {
                    pos += 2;
                    if &s[pos..=pos] == "0" {
                        tokens.push_str(" 0");
                        pos += 1;
                    } else {
                        for prefix in MOD_NUM_PREFIX.iter() {
                            if s[pos..].starts_with(prefix) {
                                pos += prefix.len();
                                let width = (prefix.len() - 1) * 4;
                                let rep = &s[pos..(pos + width)];
                                pos += width;
                                let mut num = i64::from_str_radix(rep, 2)?;
                                if sign == "10" {
                                    num = -num;
                                }
                                tokens.push_str(&format!(" {}", num));
                                break;
                            }
                        }
                    }
                }
            }
        }
        parse_src(&tokens)
    }
}

trait Modulatable {
    fn modulate(&self) -> Result<String>;
}

const MOD_NUM_PREFIX: [&str; 16] = [
    "10",
    "110",
    "1110",
    "11110",
    "111110",
    "1111110",
    "11111110",
    "111111110",
    "1111111110",
    "11111111110",
    "111111111110",
    "1111111111110",
    "11111111111110",
    "111111111111110",
    "1111111111111110",
    "11111111111111110",
];

impl Modulatable for i64 {
    fn modulate(&self) -> Result<String> {
        let n = *self;
        let mut modulated = String::new();
        if n == 0 {
            return Ok("010".to_string());
        }
        if n >= 0 {
            modulated.push_str("01");
        } else {
            modulated.push_str("10");
        }

        let n: u128 = (n as i128).abs() as u128;

        for (i, prefix) in MOD_NUM_PREFIX.iter().enumerate() {
            let bits = (i + 1) * 4;
            if n < ((1 as u128) << bits) {
                modulated.push_str(prefix);
                let binary = format!("{:b}", n);
                modulated.push_str(&format!("{:0>width$}", binary, width = bits));
                break;
            }
        }
        Ok(modulated)
    }
}

impl Modulatable for Rc<Expr> {
    fn modulate(&self) -> Result<String> {
        match self.as_ref() {
            Nil => Ok("00".to_string()),
            Num(n) => n.modulate(),
            _ => {
                let (car, cdr) = self.destruct_cons()?.unwrap();
                let mut modulated = "11".to_string();
                modulated.push_str(&car.modulate()?);
                modulated.push_str(&cdr.modulate()?);
                Ok(modulated)
            }
        }
    }
}

enum Mod {
    Nil,
    Num(i64),
    Cons(Box<Mod>, Box<Mod>),
}

impl Mod {
    fn new(e: Rc<Expr>) -> Result<Mod> {
        match e.as_ref() {
            Ap(a, cdr, _) => match a.as_ref() {
                Ap(cons, car, _) if cons.as_ref() == &Cons => Ok(Mod::Cons(
                    Box::new(Mod::new(car.clone())?),
                    Box::new(Mod::new(cdr.clone())?),
                )),
                _ => bail!("Invalid mod"),
            },
            Num(n) => Ok(Mod::Num(*n)),
            Nil => Ok(Mod::Nil),
            _ => bail!("Invalid mod"),
        }
    }
}

impl std::fmt::Display for Mod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mod::Cons(car, cdr) => write!(f, "({}, {})", *car, *cdr),
            Mod::Num(n) => write!(f, "{}", n),
            Mod::Nil => write!(f, "nil"),
        }
    }
}

#[derive(PartialEq, Clone, Debug)]
struct InteractResult {
    flag: Rc<Expr>,
    new_state: Rc<Expr>,
    images: Rc<Expr>,
}

impl InteractResult {
    fn new(expr: Rc<Expr>) -> Result<InteractResult> {
        let list = expr.to_vec()?;
        assert_eq!(list.len(), 3);
        let mut iter = list.into_iter();
        Ok(InteractResult {
            flag: iter.next().unwrap(),
            new_state: iter.next().unwrap(),
            images: iter.next().unwrap(),
        })
    }
}

pub fn bench() -> Result<()> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("task/galaxy.txt");
    let src = std::fs::read_to_string(path)?.trim().to_string();
    let mut galaxy = Galaxy::new(&src)?;
    galaxy.bench()
}

pub(crate) struct Galaxy {
    galaxy_id: usize,
    variables: Variables,
    var_lookup_count: u64,
    var_size: u64,
    eval_count: u64,
    apply_count: u64,
}

struct Variables {
    vars: Vec<(bool, Rc<Expr>)>,
}

impl Variables {
    fn new() -> Variables {
        Variables { vars: Vec::new() }
    }

    fn insert(&mut self, id: usize, expr: Rc<Expr>) {
        if self.vars.len() <= id as usize {
            self.vars.resize((id + 1) as usize, (false, Rc::new(Nil)));
        }
        // assert_eq!(&self.vars[id], &(false, Rc::new(Nil)));
        self.vars[id] = (false, expr);
    }

    fn insert_evaluated(&mut self, id: usize, expr: Rc<Expr>) {
        self.vars[id] = (true, expr);
    }

    fn lookup(&self, id: usize) -> (bool, Rc<Expr>) {
        self.vars[id].clone()
    }

    fn insert_new(&mut self, expr: Rc<Expr>) -> usize {
        self.vars.push((false, expr));
        self.vars.len() - 1
    }
}

impl Galaxy {
    fn new_for_test(src: &str) -> Result<Galaxy> {
        Ok(Galaxy {
            galaxy_id: 1,
            variables: {
                let mut var_cache = Variables::new();
                var_cache.insert(1, parse_src(src)?);
                var_cache
            },
            var_lookup_count: 0,
            var_size: 0,
            eval_count: 0,
            apply_count: 0,
        })
    }

    pub(crate) fn new(src: &str) -> Result<Galaxy> {
        let lines = src.trim().split('\n').collect::<Vec<_>>();
        let galaxy_line_re = Regex::new(r"galaxy *= :*(\d+)$").unwrap();
        let cap = galaxy_line_re.captures(lines[lines.len() - 1]).unwrap();
        let galaxy_id: usize = cap[1].parse()?;

        let mut var_cache = Variables::new();

        let re = Regex::new(r":(\d+) *= *(.*)$").unwrap();

        for line in lines.iter().take(lines.len() - 1) {
            let cap = re.captures(line).unwrap();
            debug!("var: {}", &cap[1]);
            let id = cap[1].parse::<usize>()?;
            var_cache.insert(id, parse_src(&cap[2])?);
        }

        Ok(Galaxy {
            galaxy_id,
            variables: var_cache,
            var_lookup_count: 0,
            var_size: 0,
            eval_count: 0,
            apply_count: 0,
        })
    }

    fn bench(&mut self) -> Result<()> {
        let mut click_events = [
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (8, 4),
            (2, -8),
            (3, 6),
            (0, -14),
            (-4, 10),
            (9, -3),
            (-4, 10),
            (-2, -4),
        ]
        .iter()
        .cloned()
        .collect::<VecDeque<_>>();

        let mut state = Rc::new(Nil);
        while let Some(click) = click_events.pop_front() {
            let (x, y) = click;
            let event = ap(ap(Rc::new(Cons), Rc::new(Num(x))), Rc::new(Num(y)));
            let (new_state, images) = self.interact(state, event)?;
            error!(
                "var_cache_len(): {}, var_lookup_count: {}, var_size: {}, eval_count: {}, apply_count: {}",
                self.variables.vars.len(),
                self.var_lookup_count,
                self.var_size,
                self.eval_count,
                self.apply_count
            );
            let _screen = images.to_screen()?;
            state = new_state;
        }
        assert_eq!(
            format!("{}", Mod::new(state)?),
            "(2, ((1, (-1, nil)), (0, (nil, nil))))"
        );
        Ok(())
    }

    pub fn play(
        &mut self,
        click_receiver: ClickReceiver,
        screen_sender: ScreenSender,
    ) -> Result<()> {
        let mut click_events = [
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (8, 4),
            (2, -8),
            (3, 6),
            (0, -14),
            (-4, 10),
            (9, -3),
            (-4, 10),
            (-2, -4),
            // garaxy appears -> alien send
        ]
        .iter()
        .cloned()
        .collect::<VecDeque<_>>();

        let mut state = Rc::new(Nil);
        loop {
            let click = if let Some(click) = click_events.pop_front() {
                click
            } else {
                info!("Waiting click...");
                click_receiver.recv()?
            };
            let (x, y) = click;
            info!("Interact with {:?}", (x, y));
            let event = ap(ap(Rc::new(Cons), Rc::new(Num(x))), Rc::new(Num(y)));
            let (new_state, images) = self.interact(state, event)?;
            let screen = images.to_screen()?;
            screen_sender.send(screen)?;
            state = new_state;
        }
    }

    fn interact(&mut self, state: Rc<Expr>, event: Rc<Expr>) -> Result<(Rc<Expr>, Rc<Expr>)> {
        let expr = self.interact_galaxy(state, event)?;

        let InteractResult {
            flag,
            new_state,
            images,
        } = InteractResult::new(expr)?;
        if flag.as_ref() == &Num(0) {
            Ok((new_state, images))
        } else {
            info!("Sending to alien proxy");
            let modulated = images.modulate()?;
            debug!("modulated: {:?}", modulated);
            let alien_response = api::send(modulated)?;
            debug!("raw alien_response: {:?}", alien_response);
            let event = alien_response.trim().demodulate()?;
            debug!("demodulated alien_response: {:?}", event);
            self.interact(new_state, event)
        }
    }

    fn interact_galaxy(&mut self, state: Rc<Expr>, event: Rc<Expr>) -> Result<Rc<Expr>> {
        let expr = ap(ap(Rc::new(Expr::Var(self.galaxy_id)), state), event);
        self.eval(expr)
    }

    fn eval_galaxy(&mut self) -> Result<Rc<Expr>> {
        self.eval_var(self.galaxy_id)
    }

    fn eval_var(&mut self, id: usize) -> Result<Rc<Expr>> {
        trace!("eval var: {}", id);
        self.var_lookup_count += 1;
        let (evaluated, expr) = self.variables.lookup(id);
        self.var_size += expr.size();
        if evaluated {
            Ok(expr)
        } else {
            let res = self.eval(expr)?;
            self.variables.insert_evaluated(id, res.clone());
            Ok(res)
        }
    }

    fn eval(&mut self, mut expr: Rc<Expr>) -> Result<Rc<Expr>> {
        self.eval_count += 1;
        loop {
            match expr.as_ref() {
                Ap(left, right, false) => {
                    expr = self.apply(left.clone(), right.clone())?;
                }
                Ap(_, _, true) => {
                    return Ok(expr);
                }
                Var(n) => {
                    return self.eval_var(*n);
                }
                _ => {
                    return Ok(expr);
                }
            }
        }
    }

    fn apply(&mut self, f: Rc<Expr>, x0: Rc<Expr>) -> Result<Rc<Expr>> {
        fn evaled_ap(e0: Rc<Expr>, e1: Rc<Expr>) -> Result<Rc<Expr>> {
            Ok(Rc::new(Ap(e0, e1, true)))
        }

        fn ok(e: Expr) -> Result<Rc<Expr>> {
            Ok(Rc::new(e))
        }

        self.apply_count += 1;

        trace!("apply: f: {:?}, x0: {:?}", f, x0);
        match self.eval(f.clone())?.as_ref() {
            Num(_) => bail!("can not apply: nuber"),
            Neg => match self.eval(x0)?.as_ref() {
                Num(n) => ok(Num(-n)),
                _ => bail!("can not apply: neg"),
            },
            Inc => match self.eval(x0)?.as_ref() {
                Num(n) => ok(Num(n + 1)),
                _ => bail!("can not apply: inc"),
            },
            Dec => match self.eval(x0)?.as_ref() {
                Num(n) => ok(Num(n - 1)),
                _ => bail!("can not apply: dec"),
            },
            I => self.eval(x0),
            // ap car x2 = ap x2 t
            Car => self.apply(x0, Rc::new(T)),
            Cdr => self.apply(x0, Rc::new(F)),
            // Avoid recursion
            // Car => Ok(ap(x0, T)),
            // Cdr => Ok(ap(x0, F)),
            Nil => Ok(Rc::new(T)),
            Isnil => match self.eval(x0)?.as_ref() {
                Nil => Ok(Rc::new(T)),
                _ => Ok(Rc::new(F)),
            },
            Ap(exp, e0, _) => {
                let (e0, e1): (Rc<Expr>, Rc<Expr>) = (e0.clone(), x0); // For readability.
                match self.eval(exp.clone())?.as_ref() {
                    Add => match (self.eval(e0)?.as_ref(), self.eval(e1)?.as_ref()) {
                        (Num(n0), Num(n1)) => ok(Num(n0 + n1)),
                        _ => bail!("can not apply: add"),
                    },
                    Mul => match (self.eval(e0)?.as_ref(), self.eval(e1)?.as_ref()) {
                        (Num(n0), Num(n1)) => Ok(Rc::new(Num(n0 * n1))),
                        // _ => bail!("can not apply: mul"),
                        (x, y) => bail!("can not apply: mul: e0: {:?}, e1: {:?}", x, y),
                    },
                    Div => match (self.eval(e0)?.as_ref(), self.eval(e1)?.as_ref()) {
                        (Num(n0), Num(n1)) => ok(Num(n0 / n1)),
                        _ => bail!("can not apply: div"),
                    },
                    Eq => match (self.eval(e0)?.as_ref(), self.eval(e1)?.as_ref()) {
                        (Num(n0), Num(n1)) => {
                            if n0 == n1 {
                                ok(T)
                            } else {
                                ok(F)
                            }
                        }
                        _ => bail!("can not apply: eq"),
                    },
                    Lt => match (self.eval(e0)?.as_ref(), self.eval(e1)?.as_ref()) {
                        (Num(n0), Num(n1)) => {
                            if n0 < n1 {
                                ok(T)
                            } else {
                                ok(F)
                            }
                        }
                        _ => bail!("can not apply: lt"),
                    },
                    T => self.eval(e0),
                    F => self.eval(e1),
                    S => evaled_ap(ap(exp.clone(), e0), e1),
                    C => evaled_ap(ap(exp.clone(), e0), e1),
                    B => evaled_ap(ap(exp.clone(), e0), e1),
                    // Cons => Ok(ap(ap(Cons, e0), e1)),
                    // Eval cons earger
                    Cons => evaled_ap(ap(Rc::new(Cons), self.eval(e0)?), self.eval(e1)?),
                    Ap(exp, e, _) => {
                        let (e0, e1, e2): (Rc<Expr>, Rc<Expr>, Rc<Expr>) = (e.clone(), e0, e1);
                        match self.eval(exp.clone())?.as_ref() {
                            S => {
                                // ap ap ap s x0 x1 x2   =   ap ap x0 x2 ap x1 x2
                                // ap ap ap s add inc 1   =   3
                                let e2: Rc<Expr> = if let Ap(_, _, false) = e2.as_ref() {
                                    Rc::new(Var(self.variables.insert_new(e2)))
                                } else {
                                    e2
                                };
                                let ap_x0_x2 = ap(e0, e2.clone());
                                let ap_x1_x2 = ap(e1, e2);
                                // Ok(ap(ap_x0_x2, ap_x1_x2))
                                self.apply(ap_x0_x2, ap_x1_x2)
                            }
                            C => {
                                // ap ap ap c x0 x1 x2   =   ap ap x0 x2 x1
                                // ap ap ap c add 1 2   =   3
                                // Ok(ap(ap(e0, e2), e1))
                                self.apply(ap(e0, e2), e1)
                            }
                            B => {
                                // ap ap ap b x0 x1 x2   =   ap x0 ap x1 x2
                                // ap ap ap b inc dec x0   =   x0
                                // Ok(ap(e0, ap(e1, e2)))
                                self.apply(e0, ap(e1, e2))
                            }
                            Cons => {
                                // cons
                                // ap ap ap cons x0 x1 x2   =   ap ap x2 x0 x1
                                // self.apply(ap(e2, e0), e1)

                                // Need to avoid recursion
                                Ok(ap(ap(e2, e0), e1))
                            }
                            _ => bail!("can not apply: ap ap ap"),
                        }
                    }
                    _ => bail!("can not apply: ap ap"),
                }
            }
            _ => evaled_ap(f, x0),
        }
    }
}

pub fn eval_src(src: &str) -> Result<Rc<Expr>> {
    // let exp = parse_src(src)?;
    // eval(exp)
    let mut galaxy = Galaxy::new_for_test(src)?;
    galaxy.eval_galaxy()
}

pub fn eval_galaxy_src(src: &str) -> Result<Rc<Expr>> {
    let mut galaxy = Galaxy::new(src)?;
    galaxy.eval_galaxy()
}

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
        assert_eq!(parse_src("1")?.as_ref(), &Num(1));
        assert_eq!(parse_src("add")?.as_ref(), &Add);
        assert_eq!(
            parse_src("ap ap add 1 2")?,
            ap(ap(Rc::new(Add), Rc::new(Num(1))), Rc::new(Num(2)))
        );
        assert_eq!(
            parse_src("ap ap eq 1 2")?,
            ap(ap(Rc::new(Eq), Rc::new(Num(1))), Rc::new(Num(2)))
        );
        assert!(parse_src("add 1").is_err());
        Ok(())
    }

    #[test]
    fn eval_test() -> Result<()> {
        for (src, expr) in &[
            // add
            ("ap ap add 1 2", Num(3)),
            ("ap ap add 3 ap ap add 1 2", Num(6)),
            // eq
            ("ap ap eq 1 1", T),
            ("ap ap eq 1 2", F),
            // mul
            ("ap ap mul 2 4", Num(8)),
            ("ap ap add 3 ap ap mul 2 4", Num(11)),
            // div
            ("ap ap div 4 2", Num(2)),
            ("ap ap div 4 3", Num(1)),
            ("ap ap div 4 4", Num(1)),
            ("ap ap div 4 5", Num(0)),
            ("ap ap div 5 2", Num(2)),
            ("ap ap div 6 -2", Num(-3)),
            ("ap ap div 5 -3", Num(-1)),
            ("ap ap div -5 3", Num(-1)),
            ("ap ap div -5 -3", Num(1)),
            // lt
            ("ap ap lt 0 -1", F),
            ("ap ap lt 0 0", F),
            ("ap ap lt 0 1", T),
        ] {
            assert_eq!(eval_src(src)?.as_ref(), expr);
        }
        Ok(())
    }

    #[test]
    fn eval_unary_test() -> Result<()> {
        for (src, expr) in &[
            // neg
            ("ap neg 0", Num(0)),
            ("ap neg 1", Num(-1)),
            ("ap neg -1", Num(1)),
            ("ap ap add ap neg 1 2", Num(1)),
            // inc
            ("ap inc 0", Num(1)),
            ("ap inc 1", Num(2)),
            // dec
            ("ap dec 0", Num(-1)),
            ("ap dec 1", Num(0)),
        ] {
            assert_eq!(eval_src(src)?.as_ref(), expr);
        }
        Ok(())
    }

    #[test]
    fn eval_s_c_b_test() -> Result<()> {
        for (src, expr) in &[
            // s
            ("ap ap ap s add inc 1", Num(3)),
            ("ap ap ap s mul ap add 1 6", Num(42)),
            // c
            ("ap ap ap c add 1 2", Num(3)),
            // b
            // ap ap ap b x0 x1 x2   =   ap x0 ap x1 x2
            // ap ap ap b inc dec x0   =   x0
            ("ap ap ap b neg neg 1", Num(1)),
        ] {
            assert_eq!(eval_src(src)?.as_ref(), expr);
        }
        Ok(())
    }

    #[test]
    fn eval_t_f_i_test() -> Result<()> {
        for (src, expr) in &[
            // t
            // ap ap t x0 x1   =   x0
            // ap ap t 1 5   =   1
            // ap ap t t i   =   t
            // ap ap t t ap inc 5   =   t
            // ap ap t ap inc 5 t   =   6
            ("ap ap t 1 5", Num(1)),
            ("ap ap t t 1", T),
            ("ap ap t t ap inc 5", T),
            ("ap ap t ap inc 5 t", Num(6)),
            // f
            ("ap ap f 1 2", Num(2)),
            // i
            ("ap i 0", Num(0)),
            ("ap i i", I),
        ] {
            assert_eq!(eval_src(src)?.as_ref(), expr);
        }
        Ok(())
    }

    #[test]
    fn eval_cons_test() -> Result<()> {
        for (src, expr) in &[
            // car, cdr, cons

            // car
            // ap car ap ap cons x0 x1   =   x0
            // ap car x2   =   ap x2 t
            ("ap car ap ap cons 0 1", Num(0)),
            ("ap cdr ap ap cons 0 1", Num(1)),
            // nil
            // ap nil x0   =   t
            ("ap nil 1", T),
            // isnil
            ("ap isnil nil", T),
            ("ap isnil 1", F),
        ] {
            assert_eq!(eval_src(src)?.as_ref(), expr);
        }
        Ok(())
    }

    #[test]
    fn eval_galaxy_src_test() -> Result<()> {
        for (src, expr) in &[
            (
                ":1 = 2
    galaxy = :1",
                Num(2),
            ),
            (
                ":1 = 2
    :2 = :1
    galaxy = :2",
                Num(2),
            ),
            (
                ":1 = 2
    :2 = ap inc :1
    galaxy = :2",
                Num(3),
            ),
            (
                ":1 = 2
    :2 = ap ap add 1 :1
    galaxy = :2",
                Num(3),
            ),
            (
                ":1 = ap add 1
    :2 = ap :1 2
    galaxy = :2",
                Num(3),
            ),
        ] {
            assert_eq!(eval_galaxy_src(src)?.as_ref(), expr);
        }
        Ok(())
    }

    #[test]
    fn eval_recursive_func_test() -> Result<()> {
        // From video part2
        // https://www.youtube.com/watch?v=oU4RAEQCTDE
        let src = ":1 = ap :1 1
    :2 = ap ap t 42 :1
    galaxy = :2
    ";
        assert_eq!(eval_galaxy_src(src)?.as_ref(), &Num(42));
        Ok(())
    }

    #[test]
    fn interact_galaxy_test() -> Result<()> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("task/galaxy.txt");
        let src = std::fs::read_to_string(path)?.trim().to_string();

        let mut galaxy = Galaxy::new(&src)?;
        let res = galaxy.interact_galaxy(
            Rc::new(Nil),
            ap(ap(Rc::new(Cons), Rc::new(Num(0))), Rc::new(Num(0))),
        )?;
        assert_eq!(
            format!("{}", Mod::new(res.clone())?),
            "(0, ((0, ((0, nil), (0, (nil, nil)))), ((((-1, -3), ((0, -3), ((1, -3), ((2, -2), ((-2, -1), ((-1, -1), ((0, -1), ((3, -1), ((-3, 0), ((-1, 0), ((1, 0), ((3, 0), ((-3, 1), ((0, 1), ((1, 1), ((2, 1), ((-2, 2), ((-1, 3), ((0, 3), ((1, 3), nil)))))))))))))))))))), (((-7, -3), ((-8, -2), nil)), (nil, nil))), nil)))"
        );

        let interact_result = InteractResult::new(res)?;
        assert_eq!(interact_result.flag.as_ref(), &Num(0));
        Ok(())
    }

    #[test]
    fn into_vec_test() -> Result<()> {
        for (src, v) in &[
            ("nil", vec![]),
            ("ap ap cons 1 nil", vec![Num(1)]),
            ("ap ap cons 1 ap ap cons 2 nil", vec![Num(1), Num(2)]),
            (
                "ap ap cons 1 ap ap cons 2 ap ap cons 3 nil",
                vec![Num(1), Num(2), Num(3)],
            ),
        ] {
            assert_eq!(
                parse_src(src)?.to_vec()?,
                v.into_iter()
                    .cloned()
                    .map(|e| Rc::new(e))
                    .collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    #[test]
    fn modulate_num_test() -> Result<()> {
        assert_eq!(0.modulate()?, "010");
        assert_eq!(1.modulate()?, "01100001");
        assert_eq!(2.modulate()?, "01100010");
        assert_eq!(15.modulate()?, "01101111");
        assert_eq!(16.modulate()?, "0111000010000");
        assert_eq!(255.modulate()?, "0111011111111");
        assert_eq!(256.modulate()?, "011110000100000000");
        assert_eq!((-1).modulate()?, "10100001");
        assert_eq!((-2).modulate()?, "10100010");
        Ok(())
    }

    #[test]
    fn modulate_expr_test() -> Result<()> {
        assert_eq!(parse_src("nil")?.modulate()?, "00");
        assert_eq!(parse_src("ap ap cons nil nil")?.modulate()?, "110000");
        assert_eq!(parse_src("ap ap cons 0 nil")?.modulate()?, "1101000");
        assert_eq!(
            parse_src("ap ap cons 1 2")?.modulate()?,
            "110110000101100010"
        );
        assert_eq!(
            parse_src("ap ap cons 1 ap ap cons 2 nil")?.modulate()?,
            "1101100001110110001000"
        );
        Ok(())
    }

    #[test]
    fn demodulate_test() -> Result<()> {
        for (s, expr) in &[
            ("010", "0"),
            ("01100001", "1"),
            ("10100001", "-1"),
            ("110000", "ap ap cons nil nil"),
            ("1101100001110110001000", "ap ap cons 1 ap ap cons 2 nil"),
        ] {
            assert_eq!(s.demodulate()?, parse_src(expr)?);
        }
        Ok(())
    }
}
