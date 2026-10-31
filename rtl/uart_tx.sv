module uart_tx #(
    parameter int CLKS_PER_BIT = 32'd234
) (
    input  wire       clk,
    input  wire       rst,
    input  wire [7:0] data,
    input  wire       start,
    output reg        tx,
    output wire       busy
);

    localparam integer COUNT_WIDTH = $clog2(CLKS_PER_BIT);
    localparam [COUNT_WIDTH-1:0] LAST_COUNT =
        COUNT_WIDTH'(CLKS_PER_BIT - 32'sd1);

    reg [COUNT_WIDTH-1:0] bit_count;
    reg [3:0] bit_slot;
    reg [7:0] payload;
    reg active;

    assign busy = active;

    always @(posedge clk) begin
        if (rst) begin
            tx        <= 1'b1;
            active    <= 1'b0;
            bit_count <= {COUNT_WIDTH{1'b0}};
            bit_slot  <= 4'd0;
            payload   <= 8'h00;
        end else if (!active ||
                     ((bit_slot == 4'd9) && (bit_count == LAST_COUNT))) begin
            // Idle, or the edge completing the current stop bit.
            bit_count <= {COUNT_WIDTH{1'b0}};
            bit_slot  <= 4'd0;
            if (start) begin
                payload <= data;
                tx      <= 1'b0;
                active  <= 1'b1;
            end else begin
                tx      <= 1'b1;
                active  <= 1'b0;
            end
        end else if (bit_count == LAST_COUNT) begin
            bit_count <= {COUNT_WIDTH{1'b0}};
            bit_slot  <= bit_slot + 4'd1;
            if (bit_slot < 4'd8)
                tx <= payload[bit_slot[2:0]];
            else
                tx <= 1'b1;
        end else begin
            bit_count <= bit_count + COUNT_WIDTH'(1);
        end
    end

endmodule
