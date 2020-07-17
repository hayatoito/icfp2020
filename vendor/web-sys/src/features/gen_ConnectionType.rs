#![allow(unused_imports)]
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
#[doc = "The `ConnectionType` enum."]
#[doc = ""]
#[doc = "*This API requires the following crate features to be activated: `ConnectionType`*"]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionType {
    Cellular = "cellular",
    Bluetooth = "bluetooth",
    Ethernet = "ethernet",
    Wifi = "wifi",
    Other = "other",
    None = "none",
    Unknown = "unknown",
}
