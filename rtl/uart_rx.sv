module uart_rx #(
    parameter int CLKS_PER_BIT = 234
) (
    input  wire       clk,
    input  wire       rst,
    input  wire       rx,
    output reg [7:0]  data,
    output reg        valid
);

    localparam int COUNT_WIDTH = $clog2(CLKS_PER_BIT);
    localparam logic [COUNT_WIDTH-1:0] BIT_LAST =
        COUNT_WIDTH'(CLKS_PER_BIT - 1);
    localparam logic [COUNT_WIDTH-1:0] HALF_LAST =
        COUNT_WIDTH'((CLKS_PER_BIT / 2) - 1);

    localparam logic [2:0] IDLE       = 3'd0;
    localparam logic [2:0] START_BIT  = 3'd1;
    localparam logic [2:0] DATA_BITS  = 3'd2;
    localparam logic [2:0] STOP_BIT   = 3'd3;
    localparam logic [2:0] WAIT_HIGH  = 3'd4;

    reg rx_meta;
    reg rx_sync;
    reg [2:0] state;
    reg [COUNT_WIDTH-1:0] count;
    reg [2:0] bit_index;
    reg [7:0] receive_data;

    // Only the second synchronizer stage is used by the receiver.
    always @(posedge clk) begin
        if (rst) begin
            rx_meta <= 1'b1;
            rx_sync <= 1'b1;
        end else begin
            rx_meta <= rx;
            rx_sync <= rx_meta;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state        <= IDLE;
            count        <= {COUNT_WIDTH{1'b0}};
            bit_index    <= 3'd0;
            receive_data <= 8'h00;
            data         <= 8'h00;
            valid        <= 1'b0;
        end else begin
            valid <= 1'b0;

            case (state)
                IDLE: begin
                    count     <= {COUNT_WIDTH{1'b0}};
                    bit_index <= 3'd0;
                    if (!rx_sync)
                        state <= START_BIT;
                end

                START_BIT: begin
                    // Confirm exactly CLKS_PER_BIT/2 clocks after detection.
                    if (count == HALF_LAST) begin
                        count <= {COUNT_WIDTH{1'b0}};
                        if (rx_sync)
                            state <= IDLE;
                        else
                            state <= DATA_BITS;
                    end else begin
                        count <= count + COUNT_WIDTH'(1);
                    end
                end

                DATA_BITS: begin
                    if (count == BIT_LAST) begin
                        count <= {COUNT_WIDTH{1'b0}};
                        receive_data[bit_index] <= rx_sync;
                        if (bit_index == 3'd7) begin
                            state <= STOP_BIT;
                        end else begin
                            bit_index <= bit_index + 3'd1;
                        end
                    end else begin
                        count <= count + COUNT_WIDTH'(1);
                    end
                end

                STOP_BIT: begin
                    if (count == BIT_LAST) begin
                        count <= {COUNT_WIDTH{1'b0}};
                        if (rx_sync) begin
                            data  <= receive_data;
                            valid <= 1'b1;
                            // Re-arm immediately at the stop-bit midpoint.
                            state <= IDLE;
                        end else begin
                            state <= WAIT_HIGH;
                        end
                    end else begin
                        count <= count + COUNT_WIDTH'(1);
                    end
                end

                WAIT_HIGH: begin
                    count <= {COUNT_WIDTH{1'b0}};
                    if (rx_sync)
                        state <= IDLE;
                end

                default: begin
                    state     <= IDLE;
                    count     <= {COUNT_WIDTH{1'b0}};
                    bit_index <= 3'd0;
                end
            endcase
        end
    end

endmodule
