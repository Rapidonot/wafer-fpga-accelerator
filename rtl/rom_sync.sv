// Synchronous-read memory initialised from a $readmemh image (one word per line), with a write port so the host
// can (re)load the contents at run time instead of relying on the bitstream's block-RAM initial values.
module rom_sync #(
    parameter int DEPTH = 1024,
    parameter int AW    = 10,
    parameter int DW    = 8,
    parameter     FILE  = ""
) (
    input  logic          clk,
    input  logic [AW-1:0] addr,
    output logic [DW-1:0] rdata,
    input  logic          we,
    input  logic [AW-1:0] waddr,
    input  logic [DW-1:0] wdata
);
    logic [DW-1:0] mem [0:DEPTH-1];
    initial if (FILE != "") $readmemh(FILE, mem);
    always_ff @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        rdata <= mem[addr];
    end
endmodule
