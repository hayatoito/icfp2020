// pub use lazy_static::*;
pub use anyhow::{bail, ensure, Context, Result};
pub use log::*;
pub use std::collections::{HashMap, HashSet, VecDeque};
pub use std::path::{Path, PathBuf};
pub use std::rc::Rc;

#[cfg(test)]
mod tests {
    #[test]
    fn prelude_test_dummy() {
        assert_eq!(0, 0);
    }
}

pub fn hello() {}
