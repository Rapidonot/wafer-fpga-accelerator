// Wafer CNN core: sequential single-MAC engine (see README.md, "How it works").
//
// Everything model-specific (layer count, C/O/H/W, pool flags, ROM sizes/offsets, head bias) comes from
// rtl/gen/cnn_params.svh, written by ml/export_rtl.py. Do not hand-edit sizes here.
//
// Schedule per layer l (reads map RAM l%2, writes map RAM (l+1)%2):
//   for o: load {b,M,S}[o]; for py,px (output map): for sub (2x2 pool window, or 1): acc = b;
//          for c,ky,kx: acc += in[c][y+ky-1][x+kx-1] * w[o][c][ky][kx]  (zero outside the map)
//          best = max over sub; out[o][py][px] = requant(best);  then drain the pipeline.
// After the last layer: for c: g = sum of the final map of channel c; logit[k] += g * head_w[k][c].
//
// Issue pipeline (one MAC term per clock while in S_RUN):
//   A0 counters -> A1 address regs -> A2 RAM/ROM data -> A3 MAC acc -> A4 pool max -> requant (2) -> RAM write
module cnn_core #(
    parameter WROM_FILE = "rtl/gen/wrom.hex",
    parameter PROM_FILE = "rtl/gen/prom.hex",
    parameter HROM_FILE = "rtl/gen/hrom.hex"
) (
    input  logic         clk,
    input  logic         rst,
    // input-map loader: writes map RAM 0 while the core is idle (address = c*H*W + y*W + x)
    input  logic         ld_we,
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [15:0]  ld_addr,       // fixed protocol width; low MAP_AW bits used
    /* verilator lint_on UNUSEDSIGNAL */
    input  logic [7:0]   ld_data,
    // parameter-memory reload from the host (replaces the bitstream's initial block-RAM contents)
    input  logic [2:0]   rom_we,        // one-hot: weight ROM, parameter ROM, head ROM
    /* verilator lint_off UNUSEDSIGNAL */
    input  logic [15:0]  rom_waddr,     // fixed protocol width; low *_AW bits used
    input  logic [63:0]  rom_wdata,     // low 8 / 54 / 8 bits used
    /* verilator lint_on UNUSEDSIGNAL */
    // control
    input  logic         start,
    output logic         busy,
    output logic         done,          // one-cycle pulse; results valid from then until the next start
    output logic [255:0] logits_flat,   // logit k = logits_flat[32*k +: 32], signed
    output logic [7:0]   mask,          // bit k = (logit k > 0)
    output logic [31:0]  cycles,        // clocks from start to done of the last run
    // debug / verification: every feature-map write (unused in hardware)
    output logic         dbg_we,
    output logic [15:0]  dbg_addr,
    output logic [7:0]   dbg_data,
    output logic [3:0]   dbg_layer
);
    // The generated header also defines constants used only by the top level and testbenches.
    /* verilator lint_off UNUSEDPARAM */
    `include "cnn_params.svh"
    /* verilator lint_on UNUSEDPARAM */

    localparam int DIMW = DW - 1;            // unsigned map dimension / coordinate width
    localparam int CDW  = DW + 1;            // signed coordinate width with headroom for -1 .. 2*dim
    localparam int AW2  = MAP_AW + CDW + 2;  // signed width for the address arithmetic
    localparam int PCW  = $clog2(P_LAST + 1);
    localparam int DRAIN_CYCLES = 10;        // >= issue-to-RAM-write latency (7), see header

    // ------------------------------------------------------------------ FSM
    typedef enum logic [3:0] {
        S_IDLE, S_LAYER, S_CH_ADDR, S_CH_WAIT, S_CH_PARAM, S_RUN, S_DRAIN,
        S_HCLR, S_GAP, S_GWAIT, S_HEAD, S_HWAIT, S_DONE
    } state_t;
    state_t state;

    logic [LW-1:0] l;                       // layer index
    // current layer geometry, loaded in S_LAYER from the packed tables
    logic [CW-1:0]      cur_cm1, cur_om1;   // C-1, O-1
    logic [DIMW-1:0]    cur_h, cur_w, cur_phm1, cur_pwm1;
    logic [MAP_AW-1:0]  cur_hw;
    logic               cur_pool;
    logic [KW-1:0]      cur_k;
    logic [PROM_AW-1:0] cur_pbase;
    // per output channel
    logic signed [31:0] cur_b;
    logic [15:0]        cur_m;
    logic [5:0]         cur_s;
    logic [CW-1:0]      o;
    logic [WROM_AW-1:0] w_obase;            // weight address of [o][0][0][0]
    // loop counters
    logic [CW-1:0]      c;
    logic [1:0]         ky, kx;
    logic               dy, dx;
    logic [DIMW-1:0]    py, px;
    logic [MAP_AW-1:0]  c_base;             // c * H * W
    logic [WROM_AW-1:0] w_addr;
    logic [MAP_AW-1:0]  out_addr;
    logic [3:0]         drain_cnt;
    logic               rsel;               // map RAM read by this layer (= l[0])
    // head phase
    logic [CW-1:0]      hc;                 // channel of the final map
    logic [PCW-1:0]     gp;                 // pixel within channel
    logic [MAP_AW-1:0]  gaddr;
    logic [2:0]         hk;
    logic [HROM_AW-1:0] haddr;
    logic [G_W-1:0]     g;

    // table fields for the current layer index (32-bit fields; only the low bits are used)
    /* verilator lint_off UNUSEDSIGNAL */
    logic [FW-1:0] t_c, t_o, t_h, t_w, t_hw, t_ph, t_pw, t_pool, t_k, t_wbase, t_pbase;
    assign t_c     = L_C[l*FW +: FW];
    assign t_o     = L_O[l*FW +: FW];
    assign t_h     = L_H[l*FW +: FW];
    assign t_w     = L_W[l*FW +: FW];
    assign t_hw    = L_HW[l*FW +: FW];
    assign t_ph    = L_PH[l*FW +: FW];
    assign t_pw    = L_PW[l*FW +: FW];
    assign t_pool  = L_POOL[l*FW +: FW];
    assign t_k     = L_K[l*FW +: FW];
    assign t_wbase = L_WBASE[l*FW +: FW];
    assign t_pbase = L_PBASE[l*FW +: FW];
    /* verilator lint_on UNUSEDSIGNAL */

    // ------------------------------------------------------------------ A0: issue
    logic issue, k_last, sub_last, it_first, it_subfirst;
    logic [DIMW:0]          yy, xx;          // conv-output coordinate (pre-pool)
    logic signed [CDW-1:0]  iy, ix;          // input coordinate, may be -1 or H/W
    logic                   in_map;
    /* verilator lint_off UNUSEDSIGNAL */
    logic signed [AW2-1:0]  addr_s;          // wide for the signed arithmetic; low MAP_AW bits used
    /* verilator lint_on UNUSEDSIGNAL */

    assign issue       = (state == S_RUN);
    assign k_last      = (kx == 2'd2) && (ky == 2'd2) && (c == cur_cm1);
    assign sub_last    = !cur_pool || (dy && dx);
    assign it_first    = (c == '0) && (ky == 2'd0) && (kx == 2'd0);
    assign it_subfirst = !dy && !dx;
    assign yy = cur_pool ? {py, dy} : {1'b0, py};
    assign xx = cur_pool ? {px, dx} : {1'b0, px};
    assign iy = $signed({1'b0, yy}) + $signed({{(CDW-2){1'b0}}, ky}) - $signed({{(CDW-1){1'b0}}, 1'b1});
    assign ix = $signed({1'b0, xx}) + $signed({{(CDW-2){1'b0}}, kx}) - $signed({{(CDW-1){1'b0}}, 1'b1});
    assign in_map = !iy[CDW-1] && !ix[CDW-1] &&
                    (iy < $signed({2'b00, cur_h})) && (ix < $signed({2'b00, cur_w}));
    // c*H*W + iy*W + ix (only used when in_map)
    assign addr_s = $signed({{(AW2-MAP_AW){1'b0}}, c_base})
                  + $signed({{(AW2-CDW){iy[CDW-1]}}, iy}) * $signed({{(AW2-DIMW){1'b0}}, cur_w})
                  + $signed({{(AW2-CDW){ix[CDW-1]}}, ix});

    // ------------------------------------------------------------------ A1: address registers
    logic               a1_v, a1_first, a1_last, a1_subfirst, a1_plast, a1_pad;
    logic [MAP_AW-1:0]  a1_oaddr;
    logic [MAP_AW-1:0]  ram_raddr;
    logic [WROM_AW-1:0] wrom_addr;
    // A2: memory outputs valid
    logic               a2_v, a2_first, a2_last, a2_subfirst, a2_plast, a2_pad;
    logic [MAP_AW-1:0]  a2_oaddr;
    logic [7:0]         ram_rdata, ram0_rdata, ram1_rdata;
    logic [7:0]         wrom_data;
    logic [7:0]         act;
    // A3: MAC
    logic signed [31:0] mac_acc_q;
    logic               mac_ov;
    logic               a3_subfirst, a3_plast;
    logic [MAP_AW-1:0]  a3_oaddr;
    // A4: pool
    logic signed [31:0] best, pooled;
    logic               a4_v;
    logic signed [31:0] a4_acc;
    logic [MAP_AW-1:0]  a4_oaddr;
    // requant
    logic               rq_ov;
    logic [7:0]         rq_q;
    logic [MAP_AW-1:0]  rq_oaddr1, rq_oaddr2;

    always_ff @(posedge clk) begin
        a1_v        <= issue;
        a1_first    <= it_first;
        a1_last     <= k_last;
        a1_subfirst <= it_subfirst;
        a1_plast    <= k_last && sub_last;
        a1_pad      <= !in_map;
        a1_oaddr    <= out_addr;
        wrom_addr   <= w_addr;
        if (state == S_GAP) ram_raddr <= gaddr;
        else                ram_raddr <= in_map ? addr_s[MAP_AW-1:0] : '0;

        a2_v        <= a1_v;
        a2_first    <= a1_first;
        a2_last     <= a1_last;
        a2_subfirst <= a1_subfirst;
        a2_plast    <= a1_plast;
        a2_pad      <= a1_pad;
        a2_oaddr    <= a1_oaddr;

        if (a2_v && a2_last) begin
            a3_subfirst <= a2_subfirst;
            a3_plast    <= a2_plast;
            a3_oaddr    <= a2_oaddr;
        end

        if (mac_ov) best <= pooled;
        a4_v     <= mac_ov && a3_plast;
        a4_acc   <= pooled;
        a4_oaddr <= a3_oaddr;

        rq_oaddr1 <= a4_oaddr;
        rq_oaddr2 <= rq_oaddr1;
        if (rst) begin
            a1_v <= 1'b0;
            a2_v <= 1'b0;
            a4_v <= 1'b0;
        end
    end

    assign ram_rdata = rsel ? ram1_rdata : ram0_rdata;
    assign act       = a2_pad ? 8'd0 : ram_rdata;
    assign pooled    = (a3_subfirst || (mac_acc_q > best)) ? mac_acc_q : best;

    logic signed [7:0] wgt_s;
    assign wgt_s = wrom_data;
    mac_acc u_mac (
        .clk(clk), .rst(rst), .in_valid(a2_v), .in_first(a2_first), .in_last(a2_last),
        .act(act), .wgt(wgt_s), .bias(cur_b), .acc(mac_acc_q), .out_valid(mac_ov)
    );

    requant u_rq (
        .clk(clk), .rst(rst), .in_valid(a4_v), .acc(a4_acc), .m(cur_m), .s(cur_s),
        .out_valid(rq_ov), .q(rq_q)
    );

    // ------------------------------------------------------------------ memories
    logic               ram0_we, ram1_we;
    logic [MAP_AW-1:0]  ram0_waddr;
    logic [7:0]         ram0_wdata;
    // RAM 0 is written by the loader while idle, otherwise by layers with an odd index (l+1 even).
    assign ram0_we    = busy ? (rq_ov && rsel) : ld_we;
    assign ram0_waddr = busy ? rq_oaddr2 : ld_addr[MAP_AW-1:0];
    assign ram0_wdata = busy ? rq_q : ld_data;
    assign ram1_we    = busy && rq_ov && !rsel;

    ram_sdp #(.DEPTH(MAP_DEPTH), .AW(MAP_AW), .DW(8), .SW(9)) u_ram0 (
        .clk(clk), .we(ram0_we), .waddr(ram0_waddr), .wdata(ram0_wdata), .raddr(ram_raddr), .rdata(ram0_rdata));
    ram_sdp #(.DEPTH(MAP_DEPTH), .AW(MAP_AW), .DW(8), .SW(9)) u_ram1 (
        .clk(clk), .we(ram1_we), .waddr(rq_oaddr2), .wdata(rq_q), .raddr(ram_raddr), .rdata(ram1_rdata));

    rom_sync #(.DEPTH(WROM_DEPTH), .AW(WROM_AW), .DW(8), .FILE(WROM_FILE)) u_wrom (
        .clk(clk), .addr(wrom_addr), .rdata(wrom_data),
        .we(rom_we[0]), .waddr(rom_waddr[WROM_AW-1:0]), .wdata(rom_wdata[7:0]));

    logic [PROM_AW-1:0] prom_addr;
    logic [53:0]        prom_data;            // {S[5:0], M[15:0], b[31:0]}
    rom_sync #(.DEPTH(PROM_DEPTH), .AW(PROM_AW), .DW(54), .FILE(PROM_FILE)) u_prom (
        .clk(clk), .addr(prom_addr), .rdata(prom_data),
        .we(rom_we[1]), .waddr(rom_waddr[PROM_AW-1:0]), .wdata(rom_wdata[53:0]));

    logic [7:0] hrom_data;
    rom_sync #(.DEPTH(HROM_DEPTH), .AW(HROM_AW), .DW(8), .FILE(HROM_FILE)) u_hrom (
        .clk(clk), .addr(haddr), .rdata(hrom_data),
        .we(rom_we[2]), .waddr(rom_waddr[HROM_AW-1:0]), .wdata(rom_wdata[7:0]));

    // ------------------------------------------------------------------ head
    logic       h_clear, h_en, gv1, gv2, hv1;
    logic [2:0] hk1;
    logic signed [7:0] hw_s;
    assign hw_s = hrom_data;
    head_unit #(.GW(G_W)) u_head (
        .clk(clk), .rst(rst), .clear(h_clear), .bias_flat(HEAD_B), .en(hv1), .k(hk1), .g(g),
        .w(hw_s), .logits_flat(logits_flat), .mask(mask));
    assign h_clear = (state == S_HCLR);
    assign h_en    = (state == S_HEAD);

    always_ff @(posedge clk) begin
        gv1    <= (state == S_GAP);
        gv2    <= gv1;
        hv1    <= h_en;
        hk1    <= hk;
        if (state == S_HCLR || (state == S_HWAIT)) g <= '0;
        else if (gv2) g <= g + G_W'(ram_rdata);
        if (rst) begin
            gv1 <= 1'b0; gv2 <= 1'b0; hv1 <= 1'b0;
        end
    end

    // ------------------------------------------------------------------ control
    assign busy = (state != S_IDLE);
    assign dbg_we    = busy && rq_ov;
    assign dbg_addr  = 16'(rq_oaddr2);
    assign dbg_data  = rq_q;
    assign dbg_layer = 4'(l);

    always_ff @(posedge clk) begin
        done <= 1'b0;
        if (busy) cycles <= cycles + 32'd1;
        case (state)
            S_IDLE: if (start) begin
                l      <= '0;
                cycles <= 32'd1;
                state  <= S_LAYER;
            end
            S_LAYER: begin
                cur_cm1   <= t_c[CW-1:0] - 1'b1;
                cur_om1   <= t_o[CW-1:0] - 1'b1;
                cur_h     <= t_h[DIMW-1:0];
                cur_w     <= t_w[DIMW-1:0];
                cur_hw    <= t_hw[MAP_AW-1:0];
                cur_phm1  <= t_ph[DIMW-1:0] - 1'b1;
                cur_pwm1  <= t_pw[DIMW-1:0] - 1'b1;
                cur_pool  <= t_pool[0];
                cur_k     <= t_k[KW-1:0];
                cur_pbase <= t_pbase[PROM_AW-1:0];
                w_obase   <= t_wbase[WROM_AW-1:0];
                rsel      <= l[0];
                o         <= '0;
                out_addr  <= '0;
                state     <= S_CH_ADDR;
            end
            S_CH_ADDR: begin
                prom_addr <= cur_pbase + PROM_AW'(o);
                c <= '0; ky <= '0; kx <= '0; dy <= 1'b0; dx <= 1'b0; py <= '0; px <= '0;
                c_base <= '0;
                w_addr <= w_obase;
                state  <= S_CH_WAIT;
            end
            S_CH_WAIT: state <= S_CH_PARAM;   // ROM samples prom_addr on this edge
            S_CH_PARAM: begin
                cur_b <= $signed(prom_data[31:0]);
                cur_m <= prom_data[47:32];
                cur_s <= prom_data[53:48];
                state <= S_RUN;
            end
            S_RUN: begin
                // advance kx -> ky -> c -> sub-window -> px -> py
                w_addr <= k_last ? w_obase : w_addr + 1'b1;
                if (kx != 2'd2) kx <= kx + 1'b1;
                else begin
                    kx <= '0;
                    if (ky != 2'd2) ky <= ky + 1'b1;
                    else begin
                        ky <= '0;
                        if (c != cur_cm1) begin
                            c      <= c + 1'b1;
                            c_base <= c_base + cur_hw;
                        end else begin
                            c      <= '0;
                            c_base <= '0;
                            if (!sub_last) begin
                                if (!dx) dx <= 1'b1;
                                else begin dx <= 1'b0; dy <= 1'b1; end
                            end else begin
                                dx <= 1'b0;
                                dy <= 1'b0;
                                out_addr <= out_addr + 1'b1;
                                if (px != cur_pwm1) px <= px + 1'b1;
                                else begin
                                    px <= '0;
                                    if (py != cur_phm1) py <= py + 1'b1;
                                    else begin
                                        py        <= '0;
                                        drain_cnt <= 4'(DRAIN_CYCLES);
                                        state     <= S_DRAIN;
                                    end
                                end
                            end
                        end
                    end
                end
            end
            S_DRAIN: begin
                // let the last terms flow through MAC, pool, requant and into the RAM
                if (drain_cnt != 0) drain_cnt <= drain_cnt - 1'b1;
                else if (o != cur_om1) begin
                    o       <= o + 1'b1;
                    w_obase <= w_obase + WROM_AW'(cur_k);
                    state   <= S_CH_ADDR;
                end else if (l != LW'(NL - 1)) begin
                    l     <= l + 1'b1;
                    state <= S_LAYER;
                end else begin
                    // final map is in RAM NL%2
                    rsel  <= (NL % 2) == 1;
                    state <= S_HCLR;
                end
            end
            S_HCLR: begin                      // logits <= head bias, g <= 0
                hc    <= '0;
                gp    <= '0;
                gaddr <= '0;
                state <= S_GAP;
            end
            S_GAP: begin                       // read P_LAST pixels of channel hc
                gaddr <= gaddr + 1'b1;
                if (gp == PCW'(P_LAST - 1)) begin
                    gp        <= '0;
                    drain_cnt <= 4'd2;
                    state     <= S_GWAIT;
                end else gp <= gp + 1'b1;
            end
            S_GWAIT: begin                     // last pixel reaches g
                if (drain_cnt != 0) drain_cnt <= drain_cnt - 1'b1;
                else begin
                    hk    <= '0;
                    haddr <= HROM_AW'(hc) << 3;   // head ROM is [c][k]
                    state <= S_HEAD;
                end
            end
            S_HEAD: begin                      // 8 terms logit[k] += g * w[k][hc]
                haddr <= haddr + 1'b1;
                hk    <= hk + 1'b1;
                if (hk == 3'd7) begin
                    drain_cnt <= 4'd2;
                    state     <= S_HWAIT;
                end
            end
            S_HWAIT: begin                     // last head term accumulated; g cleared here
                if (drain_cnt != 0) drain_cnt <= drain_cnt - 1'b1;
                else if (hc != CW'(C_LAST - 1)) begin
                    hc    <= hc + 1'b1;
                    state <= S_GAP;
                end else state <= S_DONE;
            end
            S_DONE: begin
                done  <= 1'b1;
                state <= S_IDLE;
            end
            default: state <= S_IDLE;
        endcase
        if (rst) begin
            state  <= S_IDLE;
            done   <= 1'b0;
            cycles <= '0;
            l      <= '0;
        end
    end
endmodule
