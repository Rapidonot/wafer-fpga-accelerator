// Tang Nano 9K top level: host protocol (README.md, "Protocol") around cnn_core.
//
// Host -> FPGA : 0xA5, then IN_BYTES input counts (plane-major, row-major). Other bytes while idle are ignored.
// FPGA -> host : 0x5A, mask, 8 x int32 logits (little endian), uint32 compute cycles (little endian) = 38 bytes.
// Host -> FPGA : 0xA9, sel, then the whole image of one parameter memory (sel 0: weight ROM, 1 B per entry;
//                1: parameter ROM, 7 B per entry, little endian, low 54 bits used; 2: head ROM, 1 B per entry).
// FPGA -> host : 0x5E, sel once the image is written. This replaces the bitstream's initial block-RAM contents.
// If the host stops sending in the middle of an input map or image for RX_TIMEOUT clocks, the partial transfer
// is dropped and the receiver goes back to waiting for a command (0 disables the timeout).
// LEDs (active low): led_n[0] busy (loading/computing/sending), led_n[1] result ready,
//                    led_n[5:2] defect mask bits 0..3 (Center, Donut, Edge-Loc, Edge-Ring) of the last result.
// Reset: power-on reset (16 clocks) or button S1 (active low), synchronised; active-high synchronous inside.
module wafer_top #(
    parameter int CLKS_PER_BIT = 234,         // 27 MHz / 115200 baud = 234.375 (-0.16 %)
    parameter int RX_TIMEOUT   = 2_700_000,   // 100 ms at 27 MHz
    parameter     WROM_FILE    = "rtl/gen/wrom.hex",
    parameter     PROM_FILE    = "rtl/gen/prom.hex",
    parameter     HROM_FILE    = "rtl/gen/hrom.hex"
) (
    input  logic       clk,        // 27 MHz crystal, pin 52
    input  logic       btn_s1_n,   // S1, pin 3, low when pressed
    input  logic       uart_rx,    // pin 18, from the BL702 USB-UART
    output logic       uart_tx,    // pin 17, to the BL702 USB-UART
    output logic [5:0] led_n       // pins 10, 11, 13, 14, 15, 16, active low
);
    /* verilator lint_off UNUSEDPARAM */
    `include "cnn_params.svh"
    /* verilator lint_on UNUSEDPARAM */

    localparam int TOW = (RX_TIMEOUT > 1) ? $clog2(RX_TIMEOUT + 1) : 1;

    // ------------------------------------------------------------------ reset
    // Power-up values are FPGA flip-flop init values (configuration), not a reset; hence the waiver.
    /* verilator lint_off PROCASSINIT */
    logic       btn_meta = 1'b1, btn_sync = 1'b1;
    logic [3:0] por_cnt = 4'd0;
    logic       rst = 1'b1;
    /* verilator lint_on PROCASSINIT */
    always_ff @(posedge clk) begin
        btn_meta <= btn_s1_n;
        btn_sync <= btn_meta;
        if (por_cnt != 4'hF) por_cnt <= por_cnt + 4'd1;
        rst <= (por_cnt != 4'hF) || !btn_sync;
    end

    // ------------------------------------------------------------------ UART
    logic [7:0] rx_data, tx_data;
    logic       rx_valid, tx_start, tx_busy;
    uart_rx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_rx (
        .clk(clk), .rst(rst), .rx(uart_rx), .data(rx_data), .valid(rx_valid));
    uart_tx #(.CLKS_PER_BIT(CLKS_PER_BIT)) u_tx (
        .clk(clk), .rst(rst), .data(tx_data), .start(tx_start), .tx(uart_tx), .busy(tx_busy));

    // ------------------------------------------------------------------ core
    logic         ld_we, core_start, core_done;
    logic [15:0]  ld_addr;
    logic [7:0]   ld_data;
    logic [255:0] logits_flat;
    logic [7:0]   mask;
    logic [31:0]  cycles;
    /* verilator lint_off UNUSEDSIGNAL */
    logic         core_busy;        // the protocol FSM state already covers it
    logic         dbg_we;
    logic [15:0]  dbg_addr;
    logic [7:0]   dbg_data;
    logic [3:0]   dbg_layer;
    /* verilator lint_on UNUSEDSIGNAL */
    logic [2:0]  rom_we;
    logic [15:0] rom_waddr;
    logic [63:0] rom_wdata;
    cnn_core #(.WROM_FILE(WROM_FILE), .PROM_FILE(PROM_FILE), .HROM_FILE(HROM_FILE)) u_core (
        .clk(clk), .rst(rst), .ld_we(ld_we), .ld_addr(ld_addr), .ld_data(ld_data),
        .rom_we(rom_we), .rom_waddr(rom_waddr), .rom_wdata(rom_wdata), .start(core_start),
        .busy(core_busy), .done(core_done), .logits_flat(logits_flat), .mask(mask), .cycles(cycles),
        .dbg_we(dbg_we), .dbg_addr(dbg_addr), .dbg_data(dbg_data), .dbg_layer(dbg_layer));

    // ------------------------------------------------------------------ protocol FSM
    typedef enum logic [2:0] { P_IDLE, P_LOAD, P_START, P_RUN, P_SEND, P_FLUSH, P_RSEL, P_RLOAD } pstate_t;
    pstate_t     ps;
    logic [15:0] cnt;               // bytes loaded / bytes sent
    logic [TOW-1:0] idle_cnt;       // clocks since the last received byte while loading
    logic [303:0] resp;             // {cycles, logits, mask, 0x5A}, byte i = resp[8*i +: 8]
    logic [5:0]  resp_last;         // index of the last response byte (37 for a result, 1 for an image ack)
    logic [1:0]  rsel;              // memory being reloaded
    logic [2:0]  rbyte;             // byte index within the current entry
    logic [63:0] rword;             // entry being assembled
    logic        rwrite;            // write rword at rom_waddr this cycle
    logic        result_ready;
    logic [3:0]  led_mask;

    assign ld_we      = (ps == P_LOAD) && rx_valid;
    assign ld_addr    = cnt;
    assign ld_data    = rx_data;
    assign core_start = (ps == P_START);
    assign tx_start   = (ps == P_SEND) && !tx_busy;
    assign tx_data    = resp[cnt[5:0]*8 +: 8];
    assign rom_we     = rwrite ? (3'b001 << rsel) : 3'b000;
    assign rom_wdata  = rword;

    // bytes per entry and entries per image of the memory being reloaded
    logic [2:0]  rbpe;
    logic [15:0] rentries;
    assign rbpe     = (rsel == 2'd1) ? 3'd7 : 3'd1;
    assign rentries = (rsel == 2'd0) ? 16'(WROM_DEPTH) : (rsel == 2'd1) ? 16'(PROM_DEPTH) : 16'(HROM_DEPTH);

    always_ff @(posedge clk) begin
        case (ps)
            P_IDLE: if (rx_valid && rx_data == 8'hA5) begin
                cnt          <= '0;
                idle_cnt     <= '0;
                result_ready <= 1'b0;
                ps           <= P_LOAD;
            end else if (rx_valid && rx_data == 8'hA9) begin
                idle_cnt     <= '0;
                ps           <= P_RSEL;
            end
            P_RSEL: begin
                if (rx_valid) begin
                    rsel     <= rx_data[1:0];
                    cnt      <= '0;
                    rbyte    <= '0;
                    rword    <= '0;
                    idle_cnt <= '0;
                    if (rx_data < 8'd3) ps <= P_RLOAD; else ps <= P_IDLE;
                end else if (RX_TIMEOUT > 0) begin
                    idle_cnt <= idle_cnt + 1'b1;
                    if (idle_cnt == TOW'(RX_TIMEOUT)) ps <= P_IDLE;
                end
            end
            P_RLOAD: begin
                if (rwrite) begin              // the entry assembled on the previous edge was written
                    rword <= '0;
                    cnt   <= cnt + 16'd1;
                    if (cnt == rentries - 16'd1) begin
                        resp      <= {288'd0, 6'd0, rsel, 8'h5E};
                        resp_last <= 6'd1;
                        cnt       <= '0;
                        ps        <= P_SEND;
                    end
                end else if (rx_valid) begin
                    idle_cnt            <= '0;
                    rword[rbyte*8 +: 8] <= rx_data;
                    rbyte               <= (rbyte == rbpe - 3'd1) ? 3'd0 : rbyte + 3'd1;
                end else if (RX_TIMEOUT > 0) begin
                    idle_cnt <= idle_cnt + 1'b1;
                    if (idle_cnt == TOW'(RX_TIMEOUT)) ps <= P_IDLE;
                end
            end
            P_LOAD: begin
                if (rx_valid) begin
                    idle_cnt <= '0;
                    cnt      <= cnt + 16'd1;
                    if (cnt == 16'(IN_BYTES - 1)) ps <= P_START;
                end else if (RX_TIMEOUT > 0) begin
                    idle_cnt <= idle_cnt + 1'b1;
                    if (idle_cnt == TOW'(RX_TIMEOUT)) ps <= P_IDLE;
                end
            end
            P_START: ps <= P_RUN;
            P_RUN: if (core_done) begin
                resp         <= {cycles, logits_flat, mask, 8'h5A};
                resp_last    <= 6'd37;
                led_mask     <= mask[3:0];
                result_ready <= 1'b1;
                cnt          <= '0;
                ps           <= P_SEND;
            end
            P_SEND: if (!tx_busy) begin    // tx_start is high: this byte is accepted on this edge
                cnt <= cnt + 16'd1;
                if (cnt[5:0] == resp_last) ps <= P_FLUSH;
            end
            P_FLUSH: if (!tx_busy) ps <= P_IDLE;   // last byte has left
            default: ps <= P_IDLE;
        endcase
        if (rst) begin
            ps           <= P_IDLE;
            cnt          <= '0;
            result_ready <= 1'b0;
            led_mask     <= '0;
        end
    end

    // An entry is complete when its last byte arrives; it is written on the following edge.
    always_ff @(posedge clk) begin
        rwrite    <= !rst && ps == P_RLOAD && rx_valid && !rwrite && rbyte == rbpe - 3'd1;
        rom_waddr <= cnt;
    end

    assign led_n[0]   = !(ps != P_IDLE);
    assign led_n[1]   = !result_ready;
    assign led_n[5:2] = ~(led_mask & {4{result_ready}});
endmodule
