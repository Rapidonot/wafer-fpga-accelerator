#!/usr/bin/env bash
# Build the tang nano bitstream with the open source flow: yosys -> nextpnr-himbaechel -> gowin_pack.
# Usage (from the repo root): tools/build_bitstream.sh        -> build/wafer_top.fs
set -euo pipefail
cd "$(dirname "$0")/.."
[ -n "${OSS_CAD:-}" ] && export PATH="$OSS_CAD/bin:$PATH"

RTL="rtl/wafer_top.sv rtl/cnn_core.sv rtl/mac_acc.sv rtl/requant.sv rtl/head_unit.sv rtl/ram_sdp.sv
     rtl/rom_sync.sv rtl/uart_rx.sv rtl/uart_tx.sv"
mkdir -p build

yosys -q -l build/synth.log -p "read_verilog -sv -Irtl/gen $(echo $RTL); synth_gowin -top wafer_top -json build/wafer_top.json; stat"
nextpnr-himbaechel --json build/wafer_top.json --write build/wafer_top_pnr.json \
    --device GW1NR-LV9QN88PC6/I5 --vopt family=GW1N-9C --vopt cst=constraints/tang_nano_9k.cst \
    --freq 27 > build/pnr.log 2>&1 || { tail -20 build/pnr.log; exit 1; }
gowin_pack -d GW1N-9C -o build/wafer_top.fs build/wafer_top_pnr.json

grep "Max frequency" build/pnr.log | tail -1
echo "bitstream: build/wafer_top.fs"
echo "load (temporary):  openFPGALoader -b tangnano9k build/wafer_top.fs"
echo "flash (permanent): openFPGALoader -b tangnano9k -f build/wafer_top.fs"
