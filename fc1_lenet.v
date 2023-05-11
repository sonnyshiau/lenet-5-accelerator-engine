module lenet (
    input wire clk,
    input wire rst_n,

    input wire compute_start,
    output reg compute_finish,

    // Quantization scale
    input wire [31:0] scale_CONV1,
    input wire [31:0] scale_CONV2,
    input wire [31:0] scale_CONV3,
    input wire [31:0] scale_FC1,
    input wire [31:0] scale_FC2,

    // Weight sram, dual port
    output reg [ 3:0] sram_weight_wea0,
    output reg [15:0] sram_weight_addr0,
    output reg [31:0] sram_weight_wdata0,
    input wire [31:0] sram_weight_rdata0,
    output reg [ 3:0] sram_weight_wea1,
    output reg [15:0] sram_weight_addr1,
    output reg [31:0] sram_weight_wdata1,
    input wire [31:0] sram_weight_rdata1,

    // Activation sram, dual port
    output reg [ 3:0] sram_act_wea0,
    output reg [15:0] sram_act_addr0,
    output reg [31:0] sram_act_wdata0,
    input wire [31:0] sram_act_rdata0,
    output reg [ 3:0] sram_act_wea1,
    output reg [15:0] sram_act_addr1,
    output reg [31:0] sram_act_wdata1,
    input wire [31:0] sram_act_rdata1
);
    integer i;
    //state definition
    reg [3:0] state,next_state;
    parameter IDLE = 4'd0;
    parameter fc1_exe = 4'd1;
    parameter write_fc1 = 4'd2;
    parameter finish = 4'd3;
    


    //fc1
    reg fc1_address_delay;
    reg fc1_data_ready;
    wire cal_en;
    reg signed [4:0] exe_cnt;
    reg signed [4:0] exe_cnt_next;
    reg signed [31:0] fc1_p_sum;
    reg signed [15:0] fc1_mac;
    reg signed [31:0] fc1_sum;
    reg signed [31:0] relu_fc1; // sum relu result
    reg signed [31:0] fc1_scale; // sum relu * scale
    reg signed [31:0] shift_fc1; // shift scaled result
    reg signed [7:0] clip_fc1; // clip after shifting
    reg signed [7:0] fc1_out_act [83:0]; // fc1 act output 1*84
    reg [6:0] index;
    reg [6:0] index_next;
    reg [6:0] out_index;
    reg [6:0] out_index_next;

    //state update
    always @(posedge clk or negedge rst_n) 
        if(~rst_n)
            state <= IDLE;
        else
            state <= next_state;
    

    //fsm state transition
    always @*begin
        case(state)
          IDLE:
            begin
                if(compute_start)
                   next_state = fc1_exe;
                else
                   next_state = IDLE;
            end
          fc1_exe:
            begin
                if (index == 7'd84)
                   next_state = write_fc1;
                else
                   next_state = fc1_exe;
            end 
          write_fc1:
            begin
                if (out_index == 7'd88)
                   next_state = finish;
                else
                   next_state = write_fc1;
            end
          finish:
            begin
                if (compute_finish)
                    next_state = IDLE;
                else
                    next_state = finish;
            end
          default: 
            begin
                next_state = IDLE;
            end
        endcase
    end

    //load weights
    always @ (posedge clk or negedge rst_n)begin
        if (~rst_n) 
        begin
             sram_weight_wea0 <= 4'b0000; 
             sram_weight_addr0 <= 13018; //weight 初始值
             sram_weight_wea1 <= 4'b0000;
             sram_weight_addr1 <= 13019;
        end
        else if (state == fc1_exe)
        begin
            if (sram_weight_addr0 >= 15538 ) //完成fc1的weight loading
            begin
                sram_weight_wea0 <= 4'b0000;
                sram_weight_addr0 <= 15540;
                sram_weight_wea1 <= 4'b0000;
                sram_weight_addr1 <= 15541;  
            end
            else 
            begin
                sram_weight_addr0 <= sram_weight_addr0 + 16'd2; //每個addr+2讀取下一個addr0的位置
                sram_weight_addr1 <= sram_weight_addr1 + 16'd2;    
            end
        end
        
    end

    //load activation
    always @(posedge clk or negedge rst_n)begin
        if (~rst_n)
        begin
            sram_act_wea0 <= 4'b0000;  
            sram_act_addr0 <= 690; //activation initial value
            sram_act_wea1 <= 4'b0000;
            sram_act_addr1 <= 691; //activation initial value
            fc1_data_ready <= 0;//用來delay one clk
            fc1_address_delay <= 0;
        end
        else  
        begin
            if (state == fc1_exe)
            begin //692-721 get c3_activation
                if (sram_act_addr0 == 720 )begin  //完成一個channel並回去再reuse一輪做點乘
                    sram_act_addr0 <= 692; 
                    sram_act_addr1 <= 693;
                end
                else begin
                    sram_act_addr0 <= sram_act_addr0 +2;
                    sram_act_addr1 <= sram_act_addr1 +2;
                end
                if(~fc1_data_ready)
                begin
                    fc1_data_ready <= 1;
                end
            end
            else if (state == write_fc1)
            begin 
                if (~fc1_address_delay)
                begin
                    fc1_address_delay <=1;
                    sram_act_wea0 <= 4'b1111;
                    sram_act_addr0 <= 722;
                    sram_act_wea1 <= 4'b1111;
                    sram_act_addr1 <= 723;
                end
                else 
                begin
                    sram_act_addr0 <= sram_act_addr0 +2;
                    sram_act_addr1 <= sram_act_addr1 +2; 
                end
            end
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (~rst_n)
        begin
            compute_finish <= 0;
        end
        else if (state == finish)
        begin
            compute_finish <= ~compute_finish;
        end 
    end

    


    

    /////////////////////////////////////////////// 
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n)
        begin
            index <= 0; //84
        end
        else 
        begin
            index <= index_next;
        end
    end
  
    //exe_cnt
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n)
        begin
            exe_cnt <= -1; 
        end
        else 
        begin
            exe_cnt <= exe_cnt_next;
        end
    end

    always @*begin 
        if(state == fc1_exe && exe_cnt != 14)
            begin
                exe_cnt_next = exe_cnt + 1;
                index_next = index_next;
            end
        else if (state == fc1_exe && exe_cnt == 14)
            begin
                exe_cnt_next = 0;
                index_next = index + 1;
            end
        else 
        begin
            index_next = 0;
            exe_cnt_next = -1;
        end                    
    end

    assign cal_en = (exe_cnt  > -1)?1:0;
    /////////////////////////////////////////////// 
    always @(posedge clk or negedge rst_n) begin
        if(~rst_n)
        begin
            fc1_p_sum <= 0;
            for(i=0;i<84;i=i+1)
            begin
               fc1_out_act[i] <= 0;
            end    
        end
        else if (state == fc1_exe && exe_cnt != 14)
        begin
            fc1_p_sum <= fc1_p_sum + fc1_mac;  //fc1_mac先存在fc1_p_sum裏面 且一開始psum=0
        end
        
        else if (exe_cnt == 14) //當算到第十五次時，將psum歸零去計算新的一行，總共算84次
        begin
            fc1_p_sum <=0;
            fc1_out_act[index] <= clip_fc1[7:0];
        end
    end


    //calculation
    always @* begin
        if (cal_en)begin
            fc1_mac =  {{24{sram_act_rdata0[31]}},sram_act_rdata0[31:24]} * {{24{sram_weight_rdata0[31]}},sram_weight_rdata0[31:24]}
                      + {{24{sram_act_rdata0[23]}},sram_act_rdata0[23:16]} * {{24{sram_weight_rdata0[23]}},sram_weight_rdata0[23:16]}
                      + {{24{sram_act_rdata0[15]}},sram_act_rdata0[15:8]} * {{24{sram_weight_rdata0[15]}},sram_weight_rdata0[15:8]}
                      + {{24{sram_act_rdata0[7]}},sram_act_rdata0[7:0]} * {{24{sram_weight_rdata0[7]}},sram_weight_rdata0[7:0]} //4個8bit 乘上4個8bit的weight 得出第一個8-BIT的點，也就是一個act_addr * weight_addr
                      + {{24{sram_act_rdata1[31]}},sram_act_rdata1[31:24]} * {{24{sram_weight_rdata1[31]}},sram_weight_rdata1[31:24]}
                      + {{24{sram_act_rdata1[23]}},sram_act_rdata1[23:16]} * {{24{sram_weight_rdata1[23]}},sram_weight_rdata1[23:16]}
                      + {{24{sram_act_rdata1[15]}},sram_act_rdata1[15:8]} * {{24{sram_weight_rdata1[15]}},sram_weight_rdata1[15:8]}
                      + {{24{sram_act_rdata1[7]}},sram_act_rdata1[7:0]} * {{24{sram_weight_rdata1[7]}},sram_weight_rdata1[7:0]}; //4個8bit 乘上4個8bit的weight 得出第二個8-BIT的點
        end
        else begin
            fc1_mac =0;
        end
    end

    always @* begin
        if (cal_en)begin
            fc1_sum = fc1_p_sum + fc1_mac;
            relu_fc1 =  ( fc1_sum[31] == 1 ) ? 0 : fc1_sum;
            fc1_scale = relu_fc1 * scale_FC1;
            shift_fc1 = fc1_scale >> 16;
            clip_fc1 = (shift_fc1 > 127)? 127: shift_fc1;  
        end
        else begin
            fc1_sum = 0;
            relu_fc1 = 0;
            fc1_scale = 0;
            shift_fc1 = 0;
            clip_fc1 = 0;
        end
    end
    
    
    //////////////////////////////////////////////////
    //out_index
    always @(posedge clk or negedge rst_n) begin
        if (~rst_n)
        begin
            out_index <= 0;    
        end
        else begin
            out_index <= out_index_next;
        end
    end

    always @*begin
        if (state == write_fc1 )
        begin
            out_index_next = out_index + 8;  
        end
        else 
        begin
            out_index_next = 0;  
        end
    end
    //////////////////////////////////////////////////

    always @(posedge clk or negedge rst_n ) begin
        if (~rst_n)begin
            //sram_act_wea0 <= 4'b0000;
            //sram_act_wea1 <= 4'b0000;
            sram_act_wdata0 <= 1;
            sram_act_wdata1 <= 1;
        end
        else if (state == write_fc1)
            begin
                if (out_index == 7'd80)begin
                    //sram_act_wea0 <= 4'b1111;
                    //sram_act_wea1 <= 4'b1111;
                    sram_act_wdata0 <= {fc1_out_act[out_index+3], fc1_out_act[out_index+2], fc1_out_act[out_index+1], fc1_out_act[out_index]};
                end
                else if (out_index != 7'd80)begin
                    //sram_act_wea0 <= 4'b1111;
                    //sram_act_wea1 <= 4'b1111;
                    sram_act_wdata0 <= {fc1_out_act[out_index+3], fc1_out_act[out_index+2], fc1_out_act[out_index+1], fc1_out_act[out_index]};
                    sram_act_wdata1 <= {fc1_out_act[out_index+7], fc1_out_act[out_index+6], fc1_out_act[out_index+5], fc1_out_act[out_index+4]};
                end
            end
    end



    //Write back sram
reg c1_out_flag;
reg [3:0] out_index;

always @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        sram_act_wdata0 <= 1;
        sram_act_wdata1 <= 1;
        c1_out_cnt <= 1;
        out_index <= 0;
        c1_out_flag <= 0;
    end
    else if(state == W_C1 && c1_out_cnt == 1)begin
        if(c1_out_flag == 0)begin
            sram_act_wdata0 <= {c1_out1[3][out_index], c1_out1[2][out_index], c1_out1[1][out_index], c1_out1[0][out_index]};
            sram_act_wdata1 <= {c1_out1[7][out_index], c1_out1[6][out_index], c1_out1[5][out_index], c1_out1[4][out_index]};

            out_index <= out_index;
            c1_out_flag <= 1;
        end
        else if(c1_out_flag == 1)begin
            if(out_index == 13)begin
                sram_act_wdata0 <= {c1_out1[11][out_index], c1_out1[10][out_index], c1_out1[9][out_index], c1_out1[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out1[13][out_index], c1_out1[12][out_index]};


                out_index <= 0;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt + 1;
            end
            else begin
                sram_act_wdata0 <= {c1_out1[11][out_index], c1_out1[10][out_index], c1_out1[9][out_index], c1_out1[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out1[13][out_index], c1_out1[12][out_index]};

                out_index <= out_index + 1;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt;
            end 
        end
    end
    else if(state == W_C1 && c1_out_cnt == 2)begin
        if(c1_out_flag == 0)begin
            sram_act_wdata0 <= {c1_out2[3][out_index], c1_out2[2][out_index], c1_out2[1][out_index], c1_out2[0][out_index]};
            sram_act_wdata1 <= {c1_out2[7][out_index], c1_out2[6][out_index], c1_out2[5][out_index], c1_out2[4][out_index]};

            out_index <= out_index;
            c1_out_flag <= 1;
        end
        else if(c1_out_flag == 1)begin
            if(out_index == 13)begin
                sram_act_wdata0 <= {c1_out2[11][out_index], c1_out2[10][out_index], c1_out2[9][out_index], c1_out2[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out2[13][out_index], c1_out2[12][out_index]};


                out_index <= 0;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt + 1;
            end
            else begin
                sram_act_wdata0 <= {c1_out2[11][out_index], c1_out2[10][out_index], c1_out2[9][out_index], c1_out2[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out2[13][out_index], c1_out2[12][out_index]};

                out_index <= out_index + 1;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt;
            end 
        end
    end
    else if(state == W_C1 && c1_out_cnt == 3)begin
        if(c1_out_flag == 0)begin
            sram_act_wdata0 <= {c1_out3[3][out_index], c1_out3[2][out_index], c1_out3[1][out_index], c1_out3[0][out_index]};
            sram_act_wdata1 <= {c1_out3[7][out_index], c1_out3[6][out_index], c1_out3[5][out_index], c1_out3[4][out_index]};

            out_index <= out_index;
            c1_out_flag <= 1;
        end
        else if(c1_out_flag == 1)begin
            if(out_index == 13)begin
                sram_act_wdata0 <= {c1_out3[11][out_index], c1_out3[10][out_index], c1_out3[9][out_index], c1_out3[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out3[13][out_index], c1_out3[12][out_index]};


                out_index <= 0;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt + 1;
            end
            else begin
                sram_act_wdata0 <= {c1_out3[11][out_index], c1_out3[10][out_index], c1_out3[9][out_index], c1_out3[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out3[13][out_index], c1_out3[12][out_index]};

                out_index <= out_index + 1;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt;
            end 
        end
    end
    else if(state == W_C1 && c1_out_cnt == 4)begin
        if(c1_out_flag == 0)begin
            sram_act_wdata0 <= {c1_out4[3][out_index], c1_out4[2][out_index], c1_out4[1][out_index], c1_out4[0][out_index]};
            sram_act_wdata1 <= {c1_out4[7][out_index], c1_out4[6][out_index], c1_out4[5][out_index], c1_out4[4][out_index]};

            out_index <= out_index;
            c1_out_flag <= 1;
        end
        else if(c1_out_flag == 1)begin
            if(out_index == 13)begin
                sram_act_wdata0 <= {c1_out4[11][out_index], c1_out4[10][out_index], c1_out4[9][out_index], c1_out4[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out4[13][out_index], c1_out4[12][out_index]};


                out_index <= 0;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt + 1;
            end
            else begin
                sram_act_wdata0 <= {c1_out4[11][out_index], c1_out4[10][out_index], c1_out4[9][out_index], c1_out4[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out4[13][out_index], c1_out4[12][out_index]};

                out_index <= out_index + 1;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt;
            end 
        end
    end
    else if(state == W_C1 && c1_out_cnt == 5)begin
        if(c1_out_flag == 0)begin
            sram_act_wdata0 <= {c1_out5[3][out_index], c1_out5[2][out_index], c1_out5[1][out_index], c1_out5[0][out_index]};
            sram_act_wdata1 <= {c1_out5[7][out_index], c1_out5[6][out_index], c1_out5[5][out_index], c1_out5[4][out_index]};

            out_index <= out_index;
            c1_out_flag <= 1;
        end
        else if(c1_out_flag == 1)begin
            if(out_index == 13)begin
                sram_act_wdata0 <= {c1_out5[11][out_index], c1_out5[10][out_index], c1_out5[9][out_index], c1_out5[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out5[13][out_index], c1_out5[12][out_index]};


                out_index <= 0;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt + 1;
            end
            else begin
                sram_act_wdata0 <= {c1_out5[11][out_index], c1_out5[10][out_index], c1_out5[9][out_index], c1_out5[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out5[13][out_index], c1_out5[12][out_index]};

                out_index <= out_index + 1;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt;
            end 
        end
    end
    else if(state == W_C1 && c1_out_cnt == 6)begin
        if(c1_out_flag == 0)begin
            sram_act_wdata0 <= {c1_out6[3][out_index], c1_out6[2][out_index], c1_out6[1][out_index], c1_out6[0][out_index]};
            sram_act_wdata1 <= {c1_out6[7][out_index], c1_out6[6][out_index], c1_out6[5][out_index], c1_out6[4][out_index]};

            out_index <= out_index;
            c1_out_flag <= 1;
        end
        else if(c1_out_flag == 1)begin
            if(out_index == 13)begin
                sram_act_wdata0 <= {c1_out6[11][out_index], c1_out6[10][out_index], c1_out6[9][out_index], c1_out6[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out6[13][out_index], c1_out6[12][out_index]};


                out_index <= 0;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt + 1;
            end
            else begin
                sram_act_wdata0 <= {c1_out6[11][out_index], c1_out6[10][out_index], c1_out6[9][out_index], c1_out6[8][out_index]};
                sram_act_wdata1 <= {16'd0, c1_out6[13][out_index], c1_out6[12][out_index]};

                out_index <= out_index + 1;
                c1_out_flag <= 0;
                c1_out_cnt <= c1_out_cnt;
            end 
        end
    end
end

endmodule