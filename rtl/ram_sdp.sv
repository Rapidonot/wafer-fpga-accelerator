// Simple dual-port RAM: one synchronous write port, one synchronous read port (Gowin BSRAM friendly).
// Read-during-write to the same address returns the old data (never used by the core).
// SW > DW stores SW-bit words (data in the low DW bits). On the Tang Nano 9K the feature maps are stored 9 bits
// wide so synthesis maps them to the 9-bit block-RAM mode: in the 2-bit-wide mode yosys picked for 8-bit words,
// the board returned rare, placement-dependent bit errors that simulation does not show.
// The extra bit holds a copy of bit 0 and is folded back into the read data, so synthesis cannot remove it.
module ram_sdp #(
    parameter int DEPTH = 1024,
    parameter int AW    = 10,
    parameter int DW    = 8,
    parameter int SW    = DW
) (
    input  logic          clk,
    input  logic          we,
    input  logic [AW-1:0] waddr,
    input  logic [DW-1:0] wdata,
    input  logic [AW-1:0] raddr,
    output logic [DW-1:0] rdata
);
    logic [SW-1:0] mem [0:DEPTH-1];
    logic [SW-1:0] q;
    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= SW'({{(SW - DW){wdata[0]}}, wdata});
        q <= mem[raddr];
    end
    generate
        if (SW > DW) begin : g_wide
            assign rdata = {q[DW-1:1], q[0] & q[DW]};   // both copies of bit 0 are always equal
            /* verilator lint_off UNUSEDSIGNAL */
            logic unused = ^q[SW-1:DW];
            /* verilator lint_on UNUSEDSIGNAL */
        end else begin : g_plain
            assign rdata = q[DW-1:0];
        end
    endgenerate
endmodule
