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
// Add your design here
reg [3:0]sram_act_wea0_tmp;
reg [3:0]sram_act_wea1_tmp;
reg [15:0]sram_act_addr0_tmp;
reg [15:0]sram_act_addr1_tmp;
reg [15:0]sram_weight_addr0_tmp;
reg [15:0]sram_weight_addr1_tmp;
//*** FSM ***//
reg [3:0] state, next_state;
localparam IDLE = 4'd0;
localparam r_c1_w = 4'd1;
localparam c1_exe = 4'd2;
localparam c1_write = 4'd3;
localparam r_c2_w = 4'd4;
localparam c2_exe = 4'd5;
localparam c2_write = 4'd6;
localparam c3_exe = 4'd7;
localparam c3_write = 4'd8;
localparam fc1_exe = 4'd9;
localparam fc1_write = 4'd10;
localparam fc2_exe = 4'd11;
localparam fc2_bias = 4'd12;
localparam fc2_write = 4'd13;
localparam FINISH = 4'd15;
//----------------------------------c1-------------//
integer i,j;
reg [3:0] index; //weight counter
reg signed [7:0] weight [4:0][4:0];
reg signed [7:0] c1_ifmap [7:0][4:0];
reg read;//read weight 5*5 enable
reg [3:0] ch_cnt;//counter 0-7 and finish calculating one column
reg [3:0] c1_out_channel_cnt;//counter
reg [3:0]c1_out_channel_cnt_tmp;
reg signed [7:0] c1_cal_cycle_cnt;//count to 32 and return to 1 and renew one cycle count
wire c1_cal;
//5*5 product sum
reg signed [15:0] p1_c1 [4:0][4:0];
reg signed [15:0] p2_c1 [4:0][4:0];
reg signed [15:0] p3_c1 [4:0][4:0];
reg signed [15:0] p4_c1 [4:0][4:0];
reg signed [31:0] sum1, sum2, sum3, sum4;
//relu scale shift clamp
reg signed [31:0] relu1, relu2, relu3, relu4;
reg signed [47:0] c1_scale1, c1_scale2, c1_scale3, c1_scale4;
reg signed [47:0] c1_shift1, c1_shift2, c1_shift3, c1_shift4;
reg signed [7:0] c1_clamp1, c1_clamp2, c1_clamp3, c1_clamp4;
//Maxpool
reg signed [7:0] mp_1 [1:0][1:0];
reg signed [7:0] mp_2 [1:0][1:0];
reg mp_flag;//控制輸入的maxpooling位置
reg out;//maxpooling的enable
wire mp_out_en;//控制取出maxpooling enable
reg signed [7:0] mp1_out, mp2_out;//mp_out的8 bit buffer
//store 6*14*14 mp_out
reg [7:0] c1_map_out1 [13:0][13:0]; 
reg [7:0] c1_map_out2 [13:0][13:0];
reg [7:0] c1_map_out3 [13:0][13:0];
reg [7:0] c1_map_out4 [13:0][13:0];
reg [7:0] c1_map_out5 [13:0][13:0];
reg [7:0] c1_map_out6 [13:0][13:0];
reg ch_en;//change channel enable
reg signed [7:0] store;//counter
reg [3:0] col_write_index;//數到13完成一條col  舉例 0,1,2,3 直的讀
reg [3:0] shift1_row_cnt,shift2_row_cnt;//sft1數到12時完成一張14*14的map 多數一個讓他delay有時間賦值進去 舉例 0,1,2,3 橫的讀
//Write back sram
reg c1_out_flag_control;
reg [3:0] out_row_index;//counter
reg [3:0] out_row_index_tmp;
//----------------------------------c2-------------//
reg read_c2_weight;
reg [3:0] c2_weight_index;
reg signed [3:0] c2_channel_cnt;//0-5個channel
reg signed [7:0] c2_weight_map [4:0][4:0];
reg signed [7:0] c2_ifmap [7:0][4:0];
reg signed [7:0] c2_cal_cycle_cnt;
wire c2_cal;
//product
reg signed [15:0] p1_c2 [4:0][4:0];
reg signed [15:0] p2_c2 [4:0][4:0];
reg signed [15:0] p3_c2 [4:0][4:0];
reg signed [15:0] p4_c2 [4:0][4:0];
reg signed [31:0] c2_psum1, c2_psum2, c2_psum3, c2_psum4;
reg signed [31:0] c2_conv_temp[9:0][9:0];//暫時把算出來的值存進去，且還沒做relu,clamp等等的操作
reg signed [31:0] c2_temp_sum [9:0][9:0];//將其他channel的累加上去
//before maxpooling
reg signed [31:0]c2_temp_relu [9:0][9:0];
reg signed [31:0]c2_temp_scale[9:0][9:0];
reg signed [31:0]c2_temp_shift[9:0][9:0];
reg signed [7:0]c2_temp_clamp[9:0][9:0];
reg signed [7:0]c2_mp_out[4:0][4:0];
reg signed [3:0]c2_write_index_cnt;//因為一次寫出十個地址，代表需要一個數0-4的counter去控制寫入時機
reg signed [3:0]c2_write_cycle_cnt;//因為我的設計是一個5*8的ifmap會滑四次，所以代表一次會產生40個值需要計算上面的counter去控制寫到幾
reg [3:0]c2_cal_cycle_control;//0-9 +1 數到3停
reg [3:0]c2_cal_cycle_control_tmp;
reg [3:0]pre_mp_index;//0-9
reg [6:0]pre_mp_index_tmp;
reg [5:0]cnt_c2_out_channel;//0-16
reg [3:0]c2_ifmap_channel_cnt;//0-6
reg [3:0]c2_ifmap_channel_cnt_tmp;
//--------------------------------------------------------------------// 
//----------------------------------c3-------------//
reg [9:0] cnt_c3_out_index;//0-128
reg [9:0] cnt_c3_out_index_tmp;
reg signed [15:0]c3_mac;
reg signed [31:0]c3_psum;
reg signed [31:0]c3_sum;
reg [31:0]c3_fc_relu;
reg [31:0]c3_fc_scale;
reg [31:0]c3_fc_shift;
reg [7:0]c3_fc_clamp;
reg [7:0]c3_fc_buffer[119:0];
reg signed [8:0] c3_exe_cnt;//0-49
reg signed [8:0] c3_exe_cnt_tmp;//0-49
wire c3_cal;
reg [8:0] c3_col_index;
reg [8:0] c3_col_index_tmp;
//---------------------------fc1--------------------------//
//fc1
wire fc1_cal;
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
reg [6:0] fc1_index;
reg [6:0]fc1_index_tmp;
reg [6:0] fc1_out_index;
reg [6:0] fc1_out_index_tmp;
//----------------------------------------------------------//
reg [4:0]fc2_out_index;
reg [4:0]fc2_out_index_tmp;
wire fc2_cal;
reg fc2_add_bias_done;
reg signed [15:0]fc2_mac;
reg signed [31:0]fc2_psum;
reg signed [31:0]fc2_sum;
reg signed [31:0]fc2_out [9:0];
reg signed [15:0]fc2_bias_weight_buffer[9:0];
reg [4:0] fc2_bias_weight_index;
reg [4:0] fc2_bias_weight_index_tmp;
reg signed [4:0] fc2_exe_cnt;
reg signed [4:0] fc2_exe_cnt_tmp;
reg [4:0] fc2_index_tmp;
reg [4:0] fc2_index;
//State Update
always @(posedge clk or negedge rst_n) begin
    if(~rst_n)
        state <= IDLE;
    else
        state <= next_state;
end
//State Transition
always @* begin
    case (state)
    IDLE: begin
          if(compute_start)
            next_state = r_c1_w;
          else
            next_state = IDLE;
        end        
    r_c1_w: begin
          if(index == 4'd4)
            next_state = c1_exe;
          else
            next_state = r_c1_w;
        end
    c1_exe: begin
          if(ch_en && ch_cnt != 4'd7)
            next_state = r_c1_w;
          else if(ch_cnt == 4'd7)
            next_state = c1_write;
          else
            next_state = c1_exe;
        end
    c1_write: begin
          if(sram_act_addr1 == 16'd595)
            next_state = r_c2_w;
          else
            next_state = c1_write;
        end
    r_c2_w: begin
          if(c2_weight_index == 4'd4)
           next_state = c2_exe;
          else
           next_state = r_c2_w;
        end
    c2_exe: begin
        if(c2_cal_cycle_control == 4'd4 &&  c2_ifmap_channel_cnt != 4'd5)
        //當c2_cal_cycle=3時，代表一個channel和相對應的ifmap捲積結束，進入下一channel 的weights和下一層ifmap去做捲積
           next_state = r_c2_w;
        else if(c2_ifmap_channel_cnt == 4'd6)
        //代表算完6個channel的weight 一張map產生，進入寫的狀態
          next_state = c2_write; 
        else
           next_state = c2_exe;
        end
    c2_write: begin
        if(cnt_c2_out_channel ==  6'd16)
        //代表算完16個channel
          next_state = c3_exe;
        else if(cnt_c2_out_channel != 6'd15 && c2_write_index_cnt == 4'd4)
          next_state = r_c2_w;
        else
          next_state = c2_write;
        end
    c3_exe: begin
        if(c3_col_index == 120)
        //存滿120
          next_state = c3_write;
        else
          next_state = c3_exe;
        end
    c3_write: begin
        if (sram_act_addr1 == 16'd723)
          next_state = fc1_exe;
        else
          next_state = c3_write;
        end
    fc1_exe:begin
        if (fc1_index == 7'd84)
            next_state = fc1_write;
        else
            next_state = fc1_exe;
        end 
    fc1_write:begin
        if (sram_act_addr1 == 16'd745)
            next_state = fc2_exe;
        else
            next_state = fc1_write;
        end
    fc2_exe: begin
        if(fc2_index == 10)
          next_state = fc2_bias;
        else
          next_state = fc2_exe;
      end
    fc2_bias: begin
        if(fc2_add_bias_done)
          next_state = fc2_write;
        else
          next_state = fc2_bias;       
      end
    fc2_write: begin
        if (sram_act_addr1 == 16'd754)
          next_state = FINISH;
        else
          next_state = fc2_write;
        end 
    FINISH: begin
        next_state = FINISH;
      end
    default: begin
        next_state = IDLE;
        end
    endcase
end
//-------------------------------------------------------------------//
always @* begin
    if(state == FINISH)
      compute_finish = 1;
    else
      compute_finish = 0;
end
//===================================================================//
//                      act wea control                              //
//===================================================================//
always @(posedge clk or negedge rst_n)begin
    if (~rst_n)begin
        sram_act_wea0 <= 4'b0000;
        sram_act_wea1 <= 4'b0000;
    end
    else if (state == r_c1_w)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == c1_exe)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == c1_write)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == r_c2_w)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == c2_exe)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == c2_write)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == c3_exe)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == c3_write)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == fc1_exe)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == fc1_write)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == fc2_exe)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == fc2_bias)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else if (state == fc2_write)begin
        sram_act_wea0 <= sram_act_wea0_tmp;
        sram_act_wea1 <= sram_act_wea1_tmp;
    end
    else begin
        sram_act_wea0 <= 4'b0000;
        sram_act_wea1 <= 4'b0000;
    end
end

always@*begin
    case(state)
        r_c1_w:begin
            sram_act_wea0_tmp = 4'b0000;
            sram_act_wea1_tmp = 4'b0000;
        end
        c1_exe:begin
            sram_act_wea0_tmp = 4'b0000;
            sram_act_wea1_tmp = 4'b0000;
        end
        c1_write:begin
            sram_act_wea0_tmp = 4'b1111;
            sram_act_wea1_tmp = 4'b1111;
        end
        r_c2_w:begin
            sram_act_wea0_tmp = 4'b0000;
            sram_act_wea1_tmp = 4'b0000;
        end
        c2_exe:begin
            sram_act_wea0_tmp = 4'b0000;
            sram_act_wea1_tmp = 4'b0000;
        end
        c2_write:begin
            case(c2_write_cycle_cnt)
                4'd0:begin
                    if (c2_write_index_cnt == 4'd0)begin
                        if(cnt_c2_out_channel == 6'd16)begin
                            sram_act_wea0_tmp = 4'b0000;
                            sram_act_wea1_tmp = 4'b0000;
                        end
                        else begin
                            sram_act_wea0_tmp = 4'b1111;
                            sram_act_wea1_tmp = 4'b1111;
                        end
                    end
                    else if (c2_write_index_cnt == 4'd1)begin
                        sram_act_wea0_tmp = 4'b1111;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else if (c2_write_index_cnt == 4'd2)begin
                        sram_act_wea0_tmp = 4'b1111;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else if (c2_write_index_cnt == 4'd3)begin
                        sram_act_wea0_tmp = 4'b0001;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else begin
                        sram_act_wea0_tmp = 4'b0000;
                        sram_act_wea1_tmp = 4'b0000;
                    end
                end
                4'd1:begin
                    if (c2_write_index_cnt == 4'd0)begin
                        sram_act_wea0_tmp = 4'b1110;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else if (c2_write_index_cnt ==4'd1)begin
                        sram_act_wea0_tmp = 4'b1111;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else if (c2_write_index_cnt ==4'd2)begin
                        sram_act_wea0_tmp = 4'b1111;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else if (c2_write_index_cnt ==4'd3)begin
                        sram_act_wea0_tmp = 4'b0011;
                        sram_act_wea1_tmp = 4'b0000;
                    end
                    else begin
                        sram_act_wea0_tmp = 4'b0000;
                        sram_act_wea1_tmp = 4'b0000;
                    end
                end
                4'd2:begin
                    if (c2_write_index_cnt == 4'd0)begin
                        sram_act_wea0_tmp = 4'b1100;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else if (c2_write_index_cnt ==4'd1)begin
                        sram_act_wea0_tmp = 4'b1111;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else if (c2_write_index_cnt ==4'd2)begin
                        sram_act_wea0_tmp = 4'b1111;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else if (c2_write_index_cnt ==4'd3)begin
                        sram_act_wea0_tmp = 4'b0111;
                        sram_act_wea1_tmp = 4'b0000;
                    end
                    else begin
                        sram_act_wea0_tmp = 4'b0000;
                        sram_act_wea1_tmp = 4'b0000;
                    end
                end
                4'd3:begin
                    if (c2_write_index_cnt == 4'd0)begin
                        sram_act_wea0_tmp = 4'b1000;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else if (c2_write_index_cnt == 4'd1)begin
                        sram_act_wea0_tmp = 4'b1111;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else if (c2_write_index_cnt == 4'd2)begin
                        sram_act_wea0_tmp = 4'b1111;
                        sram_act_wea1_tmp = 4'b1111;
                    end
                    else if (c2_write_index_cnt == 4'd3)begin
                        sram_act_wea0_tmp = 4'b1111;
                        sram_act_wea1_tmp = 4'b0000;
                    end
                    else begin
                        sram_act_wea0_tmp = 4'b0000;
                        sram_act_wea1_tmp = 4'b0000;
                    end
                end
                default:begin
                    sram_act_wea0_tmp = 4'b0000;
                    sram_act_wea1_tmp = 4'b0000;
                end
            endcase
        end
        c3_exe:begin
            sram_act_wea0_tmp = 4'b0000;
            sram_act_wea1_tmp = 4'b0000;
        end
        c3_write:begin
            if(cnt_c3_out_index == 128)begin
                sram_act_wea0_tmp = 4'b0000;
                sram_act_wea1_tmp = 4'b0000; 
            end
            else begin
                sram_act_wea0_tmp = 4'b1111;
                sram_act_wea1_tmp = 4'b1111;
            end
        end
        fc1_exe:begin
            sram_act_wea0_tmp = 4'b0000;
            sram_act_wea1_tmp = 4'b0000;
        end
        fc1_write:begin
            if (fc1_out_index == 96)begin
                sram_act_wea0_tmp = 4'b0000;
                sram_act_wea1_tmp = 4'b0000;
            end
            else begin
                sram_act_wea0_tmp = 4'b1111;
                sram_act_wea1_tmp = 4'b1111;
            end
        end
        fc2_exe:begin
            sram_act_wea0_tmp = 4'b0000;
            sram_act_wea1_tmp = 4'b0000;
        end
        fc2_bias:begin
            sram_act_wea0_tmp = 4'b0000;
            sram_act_wea1_tmp = 4'b0000;
        end
        fc2_write:begin
            sram_act_wea0_tmp = 4'b1111;
            sram_act_wea1_tmp = 4'b1111;
        end
        default:begin
            sram_act_wea0_tmp = 4'b0000;
            sram_act_wea1_tmp = 4'b0000;
        end
    endcase
end
//===================================================================//
//                     weights wea control                           //
//===================================================================//
always@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        sram_weight_wea0 <= 4'b0000;
        sram_weight_wea1 <= 4'b0000;
    end
    else begin
        sram_weight_wea0 <= 4'b0000;
        sram_weight_wea1 <= 4'b0000;
    end
end
//===================================================================//
//                      act address control                          //
//===================================================================//
always @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        sram_act_addr0 <= 16'd0;
        sram_act_addr1 <= 16'd1;
    end
    else if (state == r_c1_w && index < 4'd4)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == c1_exe)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == c1_write )begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == r_c2_w && c2_weight_index < 4'd4)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == c2_exe)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == c2_write)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == c3_exe)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == c3_write)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == fc1_exe)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == fc1_write)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == fc2_exe)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == fc2_bias)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else if (state == fc2_write)begin
        sram_act_addr0 <= sram_act_addr0_tmp;
        sram_act_addr1 <= sram_act_addr1_tmp;
    end
    else begin
        sram_act_addr0 <= sram_act_addr0;
        sram_act_addr1 <= sram_act_addr1;
    end
end
//===================================================================//
//                      act address control                          //
//===================================================================//
always @*begin
    case(state)
    r_c1_w:begin
        if (index < 4'd4)begin
            sram_act_addr0_tmp = 16'd0;
            sram_act_addr1_tmp = 16'd1;
        end
        else begin
            sram_act_addr0_tmp = sram_act_addr0;
            sram_act_addr1_tmp = sram_act_addr1;
        end
    end
    c1_exe:begin
        if (sram_act_addr0 == 16'd248)begin
            sram_act_addr0_tmp = 16'd1;
            sram_act_addr1_tmp = 16'd2;
        end
        else if (sram_act_addr0 == 16'd249)begin
            sram_act_addr0_tmp = 16'd2;
            sram_act_addr1_tmp = 16'd3;
        end
        else if (sram_act_addr0 == 16'd250)begin
            sram_act_addr0_tmp = 16'd3;
            sram_act_addr1_tmp = 16'd4;
        end
        else if (sram_act_addr0 == 16'd251)begin
            sram_act_addr0_tmp = 16'd4;
            sram_act_addr1_tmp = 16'd5;
        end
        else if (sram_act_addr0 == 16'd252)begin
            sram_act_addr0_tmp = 16'd5;
            sram_act_addr1_tmp = 16'd6;
        end
        else if (sram_act_addr0 == 16'd253)begin
            sram_act_addr0_tmp = 16'd6;
            sram_act_addr1_tmp = 16'd7;
        end
        else if (sram_act_addr0 == 16'd294)begin
            sram_act_addr0_tmp = 16'd254;
            sram_act_addr1_tmp = 16'd255;
        end
        else begin
            sram_act_addr0_tmp = sram_act_addr0 + 16'd8;
            sram_act_addr1_tmp = sram_act_addr1 + 16'd8;
        end
    end
    c1_write:begin
        sram_act_addr0_tmp = sram_act_addr0 + 16'd2;
        sram_act_addr1_tmp = sram_act_addr1 + 16'd2;
    end
    r_c2_w:begin
        if (index < 4'd4)begin
            case(c2_ifmap_channel_cnt)
            4'd0:begin
                sram_act_addr0_tmp = 16'd256;
                sram_act_addr1_tmp = 16'd257;
            end
            4'd1:begin
                sram_act_addr0_tmp = 16'd312;
                sram_act_addr1_tmp = 16'd313;
            end
            4'd2:begin
                sram_act_addr0_tmp = 16'd368;
                sram_act_addr1_tmp = 16'd369;
            end
            4'd3:begin
                sram_act_addr0_tmp = 16'd424;
                sram_act_addr1_tmp = 16'd425;
            end
            4'd4:begin
                sram_act_addr0_tmp = 16'd480;
                sram_act_addr1_tmp = 16'd481;
            end
            4'd5:begin
                sram_act_addr0_tmp = 16'd536;
                sram_act_addr1_tmp = 16'd537;
            end 
            default:begin
                sram_act_addr0_tmp = 16'd0;
                sram_act_addr1_tmp = 16'd1;
            end
            endcase  
        end
        else begin
            sram_act_addr0_tmp = sram_act_addr0;
            sram_act_addr1_tmp = sram_act_addr1;
        end
    end
    c2_exe:begin
        case(sram_act_addr0) 
        //c2_channel_cnt=0
        16'd308:begin
            sram_act_addr0_tmp = 16'd257;
            sram_act_addr1_tmp = 16'd258;
        end
        16'd309:begin
            sram_act_addr0_tmp = 16'd258;
            sram_act_addr1_tmp = 16'd259;
        end
        //c2_channel_cnt=1
        16'd364:begin
            sram_act_addr0_tmp = 16'd313;
            sram_act_addr1_tmp = 16'd314;
        end
        16'd365:begin
            sram_act_addr0_tmp = 16'd314;
            sram_act_addr1_tmp = 16'd315;
        end
        //c2_channel_cnt=2
        16'd420:begin
            sram_act_addr0_tmp = 16'd369;
            sram_act_addr1_tmp = 16'd370;
        end
        16'd421:begin
            sram_act_addr0_tmp = 16'd370;
            sram_act_addr1_tmp = 16'd371;
        end
        //c2_channel_cnt=3
        16'd476:begin
            sram_act_addr0_tmp = 16'd425;
            sram_act_addr1_tmp = 16'd426;
        end
        16'd477:begin
            sram_act_addr0_tmp = 16'd426;
            sram_act_addr1_tmp = 16'd427;
        end
        //c2_channel_cnt=4
        16'd532:begin
            sram_act_addr0_tmp = 16'd481;
            sram_act_addr1_tmp = 16'd482;
        end
        16'd533:begin
            sram_act_addr0_tmp = 16'd482;
            sram_act_addr1_tmp = 16'd483;
        end
        //c2_channel_cnt=5
        16'd588:begin
            sram_act_addr0_tmp = 16'd537;
            sram_act_addr1_tmp = 16'd538;
        end
        16'd589:begin
            sram_act_addr0_tmp = 16'd538;
            sram_act_addr1_tmp = 16'd539;
        end
        16'd606:begin
            case(cnt_c2_out_channel)
            6'd0:begin
                sram_act_addr0_tmp = 16'd590;
                sram_act_addr1_tmp = 16'd591;
            end
            6'd1:begin
                sram_act_addr0_tmp = 16'd596;
                sram_act_addr1_tmp = 16'd597;
            end
            6'd2:begin
                sram_act_addr0_tmp = 16'd602;
                sram_act_addr1_tmp = 16'd603;
            end
            6'd3:begin
                sram_act_addr0_tmp = 16'd608;
                sram_act_addr1_tmp = 16'd609;
            end
            6'd4:begin
                sram_act_addr0_tmp = 16'd615;
                sram_act_addr1_tmp = 16'd616;
            end
            6'd5:begin
                sram_act_addr0_tmp = 16'd621;
                sram_act_addr1_tmp = 16'd622;
            end
            6'd6:begin
                sram_act_addr0_tmp = 16'd627;
                sram_act_addr1_tmp = 16'd628;
            end
            6'd7:begin
                sram_act_addr0_tmp = 16'd633;
                sram_act_addr1_tmp = 16'd634;
            end
            6'd8:begin
                sram_act_addr0_tmp = 16'd640;
                sram_act_addr1_tmp = 16'd641;
            end
            6'd9:begin
                sram_act_addr0_tmp = 16'd646;
                sram_act_addr1_tmp = 16'd647;
            end
            6'd10:begin
                sram_act_addr0_tmp = 16'd652;
                sram_act_addr1_tmp = 16'd653;
            end
            6'd11:begin
                sram_act_addr0_tmp = 16'd658;
                sram_act_addr1_tmp = 16'd659;
            end
            6'd12:begin
                sram_act_addr0_tmp = 16'd665;
                sram_act_addr1_tmp = 16'd666;
            end
            6'd13:begin
                sram_act_addr0_tmp = 16'd671;
                sram_act_addr1_tmp = 16'd672;
            end
            6'd14:begin
                sram_act_addr0_tmp = 16'd677;
                sram_act_addr1_tmp = 16'd678;
            end
            6'd15:begin
                sram_act_addr0_tmp = 16'd683;
                sram_act_addr1_tmp = 16'd684;
            end 
            default: begin
                sram_act_addr0_tmp = 16'd0;
                sram_act_addr1_tmp = 16'd1;
            end
            endcase 
        end
        default:begin
            sram_act_addr0_tmp = sram_act_addr0 + 16'd4;
            sram_act_addr1_tmp = sram_act_addr1 + 16'd4;
        end
        endcase
    end
    c2_write:begin
        case(c2_write_cycle_cnt)
            4'd0:begin
                case(c2_write_index_cnt)
                4'd0:begin
                    if(cnt_c2_out_channel == 6'd16)begin
                        sram_act_addr0_tmp = 16'd592;
                        sram_act_addr1_tmp = 16'd593;
                    end
                    else begin
                        sram_act_addr0_tmp = sram_act_addr0 + 16'd2;
                        sram_act_addr1_tmp = sram_act_addr1 + 16'd2;
                    end
                end
                4'd1 , 4'd2 , 4'd3 :begin
                    sram_act_addr0_tmp = sram_act_addr0 + 16'd2;
                    sram_act_addr1_tmp = sram_act_addr1 + 16'd2;
                end
                default: begin
                    sram_act_addr0_tmp = 0;
                    sram_act_addr1_tmp = 0;
                end
                endcase
            end
            4'd1 , 4'd2 , 4'd3 :begin
                if (c2_write_index_cnt <= 4'd3)begin
                    sram_act_addr0_tmp = sram_act_addr0 + 16'd2;
                    sram_act_addr1_tmp = sram_act_addr1 + 16'd2;
                end
                else begin
                    sram_act_addr0_tmp = 0;
                    sram_act_addr1_tmp = 0;
                end
            end
            default:begin
                sram_act_addr0_tmp = 0;
                sram_act_addr1_tmp = 0;
            end
        endcase
    end
    c3_exe:begin
        if(sram_act_addr0 == 690)begin
            if (c3_col_index != 119)begin
                sram_act_addr0_tmp = 16'd592;
                sram_act_addr1_tmp = 16'd593;
            end
            else if (c3_col_index == 119)begin
                sram_act_addr0_tmp = 16'd686;
                sram_act_addr1_tmp = 16'd687;
            end
            else begin
                sram_act_addr0_tmp = 0;
                sram_act_addr1_tmp = 0;
            end
        end
        else begin
            sram_act_addr0_tmp = sram_act_addr0 + 16'd2;
            sram_act_addr1_tmp = sram_act_addr1 + 16'd2;               
        end
    end
    c3_write:begin
        if(cnt_c3_out_index == 128)begin
            sram_act_addr0_tmp = 16'd692;
            sram_act_addr1_tmp = 16'd693;                  
        end
        else begin
            sram_act_addr0_tmp = sram_act_addr0 + 16'd2;
            sram_act_addr1_tmp = sram_act_addr1 + 16'd2;
        end
    end
    fc1_exe:begin
        if(sram_act_addr0 == 720)begin
            if (fc1_index != 83)begin
                sram_act_addr0_tmp = 16'd692;
                sram_act_addr1_tmp = 16'd693;
            end
            else if  (fc1_index == 83)begin
                sram_act_addr0_tmp = 16'd716;
                sram_act_addr1_tmp = 16'd717;
            end
            else begin
                sram_act_addr0_tmp = 0;
                sram_act_addr1_tmp = 0;
            end
        end
        else begin
            sram_act_addr0_tmp = sram_act_addr0 + 16'd2;
            sram_act_addr1_tmp = sram_act_addr1 + 16'd2;               
        end
    end
    fc1_write:begin
        if(fc1_out_index == 96)begin 
            sram_act_addr0_tmp = 16'd722;
            sram_act_addr1_tmp = 16'd723;                  
        end
        else begin
            sram_act_addr0_tmp = sram_act_addr0 + 16'd2;
            sram_act_addr1_tmp = sram_act_addr1 + 16'd2;
        end         
    end
    fc2_exe:begin
        if(sram_act_addr0 == 742)begin
            sram_act_addr0_tmp = 16'd722;
            sram_act_addr1_tmp = 16'd723;
        end
        else begin
            sram_act_addr0_tmp = sram_act_addr0 + 16'd2;
            sram_act_addr1_tmp = sram_act_addr1 + 16'd2;               
        end
    end
    fc2_bias:begin
        sram_act_addr0_tmp = 16'd741;
        sram_act_addr1_tmp = 16'd742;
    end
    fc2_write:begin
        sram_act_addr0_tmp = sram_act_addr0 + 16'd2;
        sram_act_addr1_tmp = sram_act_addr1 + 16'd2; 
    end
    default:begin
        sram_act_addr0_tmp = sram_act_addr0;
        sram_act_addr1_tmp = sram_act_addr1;
    end
    endcase
end
//===================================================================//
//                      weights address control                      //
//===================================================================//
always @(posedge clk or negedge rst_n) begin
    if(~rst_n) begin
        sram_weight_addr0 <= 16'd0;
        sram_weight_addr1 <= 16'd1;
    end
    else if(state == r_c1_w && index < 4'd4)begin
        sram_weight_addr0 <= sram_weight_addr0_tmp;
        sram_weight_addr1 <= sram_weight_addr1_tmp;
    end 
    else if(state == r_c2_w && c2_weight_index < 4'd4)begin
        sram_weight_addr0 <= sram_weight_addr0_tmp;
        sram_weight_addr1 <= sram_weight_addr1_tmp;
    end 
    else if (state == c3_exe)begin
        sram_weight_addr0 <= sram_weight_addr0_tmp;
        sram_weight_addr1 <= sram_weight_addr1_tmp;
    end
    else if (state == fc1_exe)begin
        sram_weight_addr0 <= sram_weight_addr0_tmp;
        sram_weight_addr1 <= sram_weight_addr1_tmp;
    end
    else if (state == fc2_exe)begin
        sram_weight_addr0 <= sram_weight_addr0_tmp;
        sram_weight_addr1 <= sram_weight_addr1_tmp;
    end
    else if (state == fc2_bias)begin
        sram_weight_addr0 <= sram_weight_addr0_tmp;
        sram_weight_addr1 <= sram_weight_addr1_tmp;
    end
    else begin
        sram_weight_addr0 <= sram_weight_addr0;
        sram_weight_addr1 <= sram_weight_addr1;
    end
end
//===================================================================//
//                      weights address control                      //
//===================================================================//
always@*begin
    if(state == r_c1_w && index < 4'd4)begin
        sram_weight_addr0_tmp = sram_weight_addr0 + 16'd2;
        sram_weight_addr1_tmp = sram_weight_addr1 + 16'd2;
    end 
    //weight address for c2
    else if(state == r_c2_w && c2_weight_index < 4'd4)begin
        sram_weight_addr0_tmp = sram_weight_addr0 + 16'd2;
        sram_weight_addr1_tmp = sram_weight_addr1 + 16'd2;
    end
    else if (state == c3_exe)begin
        if(sram_act_addr0 == 690)begin
            if (c3_col_index != 119)begin
                sram_weight_addr0_tmp = sram_weight_addr0 + 16'd2;
                sram_weight_addr1_tmp = sram_weight_addr1 + 16'd2;
            end
            else if (c3_col_index == 119)begin
                sram_weight_addr0_tmp = 16'd13016;
                sram_weight_addr1_tmp = 16'd13017;
                end
            else begin
                sram_weight_addr0_tmp = sram_weight_addr0;
                sram_weight_addr1_tmp = sram_weight_addr1;
            end
        end
        else begin
            sram_weight_addr0_tmp = sram_weight_addr0 + 16'd2;
            sram_weight_addr1_tmp = sram_weight_addr1 + 16'd2;
        end
    end
    else if (state == fc1_exe)begin
        if (sram_weight_addr0 >= 15538 ) begin
            sram_weight_addr0_tmp = 15540;
            sram_weight_addr1_tmp = 15541; 
        end
        else begin
            sram_weight_addr0_tmp = sram_weight_addr0 + 16'd2;
            sram_weight_addr1_tmp = sram_weight_addr1 + 16'd2;    
        end
    end
    else if (state == fc2_exe)begin
        if(sram_act_addr0 == 742)begin
            if(fc2_exe_cnt == 9 && fc2_index != 9)begin
                sram_weight_addr0_tmp = sram_weight_addr0 + 16'd1;
                sram_weight_addr1_tmp = sram_weight_addr1 + 16'd1;                   
            end
            else if(fc2_exe_cnt == 9 && fc2_index == 9)begin
                sram_weight_addr0_tmp = 16'd15748;
                sram_weight_addr1_tmp = 16'd15749;                       
            end
            else begin
                sram_weight_addr0_tmp = sram_weight_addr0 + 16'd2;
                sram_weight_addr1_tmp = sram_weight_addr1 + 16'd2;
            end
        end
        else begin
            sram_weight_addr0_tmp = sram_weight_addr0 + 16'd2;
            sram_weight_addr1_tmp = sram_weight_addr1 + 16'd2;
        end
    end
    else if (state == fc2_bias)begin
        sram_weight_addr0_tmp = sram_weight_addr0 + 16'd2;
        sram_weight_addr1_tmp = sram_weight_addr1 + 16'd2;
    end
    else begin
        sram_weight_addr0_tmp = sram_weight_addr0;
        sram_weight_addr1_tmp = sram_weight_addr1;
    end
end
//===================================================================//
//                      load weights map 5*5                         //
//===================================================================//
always @(posedge clk or negedge rst_n) begin
    if(~rst_n)
      read <= 0;
    else 
      read <= (state == r_c1_w) ? 1:0; 
end
always@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        index <= 4'd0;
        for (i=0; i<5; i=i+1) begin
            for (j=0; j<5; j=j+1)begin
                weight[i][j] <= 0;
            end
        end
    end
    else if(state != r_c1_w) begin
        index <= 4'd0;
        for (i=0; i<5; i=i+1) begin
            for (j=0; j<5; j=j+1)begin
                weight[i][j] <= weight[i][j];
            end
        end
    end
    else begin
        if(read == 1)begin
            index <= index + 4'd1;
            weight[0][index] <= sram_weight_rdata0[7:0];
            weight[1][index] <= sram_weight_rdata0[15:8];
            weight[2][index] <= sram_weight_rdata0[23:16];
            weight[3][index] <= sram_weight_rdata0[31:24];
            weight[4][index] <= sram_weight_rdata1[7:0];
        end
    end
end
//===================================================================//
//                      load  c1_ifmap 5*8                           //
//===================================================================//
always@(posedge clk or negedge rst_n) begin
    if(~rst_n) begin
        for (i=0; i<8; i=i+1) begin
            for (j=0; j<5; j=j+1) 
                c1_ifmap [i][j] <= 0;
        end
    end
    else if (state == c1_exe) begin
        c1_ifmap[0][4] <= sram_act_rdata0[7:0];
        c1_ifmap[1][4] <= sram_act_rdata0[15:8];
        c1_ifmap[2][4] <= sram_act_rdata0[23:16];
        c1_ifmap[3][4] <= sram_act_rdata0[31:24];
        c1_ifmap[4][4] <= sram_act_rdata1[7:0];
        c1_ifmap[5][4] <= sram_act_rdata1[15:8];
        c1_ifmap[6][4] <= sram_act_rdata1[23:16];
        c1_ifmap[7][4] <= sram_act_rdata1[31:24];
        for (i=0; i<8 ; i=i+1) begin
            for (j=0; j<4; j=j+1) begin
                c1_ifmap[i][j] <= c1_ifmap[i][j+1];
            end
        end
    end             
end
//cycle c1 counter state=c1_exe且<32 則counter +1 否則cyle=32則為1不然為-1
always @(posedge clk) begin
    c1_cal_cycle_cnt <= (state == c1_exe && c1_cal_cycle_cnt < 32) ? c1_cal_cycle_cnt + 1 : (c1_cal_cycle_cnt == 32 ? 1 : -1);
end
//MAC
assign c1_cal = (c1_cal_cycle_cnt > 4) ? 1 : 0; //if_map load up 
always @* begin
    if(c1_cal)begin
        for (i=0; i<5; i=i+1)begin
            for (j=0; j<5; j=j+1)begin
                p1_c1[i][j] = weight[i][j] * c1_ifmap[i][j];
                p2_c1[i][j] = weight[i][j] * c1_ifmap[i+1][j];
                p3_c1[i][j] = weight[i][j] * c1_ifmap[i+2][j];
                p4_c1[i][j] = weight[i][j] * c1_ifmap[i+3][j];
            end
        end
    end
    else begin
        for (i=0; i<5; i=i+1)begin
            for (j=0; j<5; j=j+1)begin
                p1_c1[i][j] = 0;
                p2_c1[i][j] = 0;
                p3_c1[i][j] = 0;
                p4_c1[i][j] = 0;
            end
        end
    end
end
//-------------------------------------------------------------------//
always @* begin
    if(c1_cal)begin
        sum1 = 0;
        sum2 = 0;
        sum3 = 0;
        sum4 = 0;
        for (i = 0; i < 5; i = i + 1) begin
            for (j = 0; j < 5; j = j + 1) begin
                sum1 = sum1 + p1_c1[i][j];
                sum2 = sum2 + p2_c1[i][j];
                sum3 = sum3 + p3_c1[i][j];
                sum4 = sum4 + p4_c1[i][j];
            end
        end
    end
    else begin
        sum1 = 0;
        sum2 = 0;
        sum3 = 0;
        sum4 = 0;
    end
end
//c1 ReLU & scale & shift & clamp 
always @* begin
    if (c1_cal)begin
        relu1 = (sum1[31] == 1) ?0:sum1;
        relu2 = (sum2[31] == 1) ?0:sum2;
        relu3 = (sum3[31] == 1) ?0:sum3;
        relu4 = (sum4[31] == 1) ?0:sum4;

        c1_scale1 = relu1 * scale_CONV1;
        c1_scale2 = relu2 * scale_CONV1;
        c1_scale3 = relu3 * scale_CONV1;
        c1_scale4 = relu4 * scale_CONV1;

        c1_shift1 = c1_scale1 >> 16;
        c1_shift2 = c1_scale2 >> 16;
        c1_shift3 = c1_scale3 >> 16;
        c1_shift4 = c1_scale4 >> 16;

        c1_clamp1 = (c1_shift1 > 127)? 127: c1_shift1; 
        c1_clamp2 = (c1_shift2 > 127)? 127: c1_shift2; 
        c1_clamp3 = (c1_shift3 > 127)? 127: c1_shift3; 
        c1_clamp4 = (c1_shift4 > 127)? 127: c1_shift4; 
    end
    else begin
        c1_scale1 = 0;
        c1_scale2 = 0;
        c1_scale3 = 0;
        c1_scale4 = 0;

        relu1 = 0;
        relu2 = 0;
        relu3 = 0;
        relu4 = 0;

        c1_shift1 = 0;
        c1_shift2 = 0;
        c1_shift3 = 0;
        c1_shift4 = 0;

        c1_clamp1 = 0; 
        c1_clamp2 = 0;  
        c1_clamp3 = 0;  
        c1_clamp4 = 0; 
    end
end
//-----------------------------------------------------------------------------//
always@(posedge clk or negedge rst_n) begin
    if(~rst_n) begin
        for(i=0; i<2; i=i+1)begin
            for(j=0; j<2; j=j+1)begin
                mp_1[i][j] <= 1;
                mp_2[i][j] <= 1;
            end
        end
        mp_flag <= 0;
        out <=0;
    end
    else begin
        out <= c1_cal ? 1:0;
        if (out)begin
            if (mp_flag)begin
                mp_1[0][0] <= c1_clamp1;
                mp_1[1][0] <= c1_clamp2;
                mp_2[0][0] <= c1_clamp3;
                mp_2[1][0] <= c1_clamp4;
                mp_flag <= 0;
            end
            else begin
                mp_1[0][1] <= c1_clamp1;
                mp_1[1][1] <= c1_clamp2;    
                mp_2[0][1] <= c1_clamp3;
                mp_2[1][1] <= c1_clamp4;
                mp_flag <= 1;
            end
        end
        else if(c1_cal_cycle_cnt == 1)begin
            mp_1[0][0] <= c1_clamp1;
            mp_1[1][0] <= c1_clamp2;
            mp_2[0][0] <= c1_clamp3;
            mp_2[1][0] <= c1_clamp4;
            mp_flag <= 0;
        end
    end
end
//----------------------------------------------------------//
assign mp_out_en = (out == 1 || c1_cal == 1) ? 1 : 0;
//maxpooling使用for迴圈去選出最大值並且輸入到mp_out中
always @* begin
    if (mp_out_en)begin
        mp1_out = 0;
        mp2_out = 0;
        for(i=0;i<2;i=i+1)begin
            for (j=0;j<2;j=j+1)begin
                if (mp_1[i][j] >= mp1_out) begin
                    mp1_out = mp_1[i][j];
                end
                if (mp_2[i][j] >= mp2_out) begin
                    mp2_out = mp_2[i][j];
                end
            end
        end
    end
    else begin
        mp1_out = 0;
        mp2_out = 0;
    end
end
//------------------------------------------------------------//
always @(posedge clk) begin
    store <= (c1_cal)?(store + 1):-2;
end
//change the channel
always @(posedge clk or negedge rst_n) begin
    if (~rst_n)begin
        ch_en <= 0;
    end
    else if(shift1_row_cnt == 14)begin
        ch_en <= 1;
    end
    else if (state == r_c1_w)begin
        ch_en <= 0;
    end
end
//store  maxpooling map to c1_out 
always@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        ch_cnt <= 1;
        for (i=0; i<14; i=i+1)begin
            for(j=0; j<14; j=j+1)begin
                c1_map_out1[i][j] <= 1;
                c1_map_out2[i][j] <= 1;
                c1_map_out3[i][j] <= 1;
                c1_map_out4[i][j] <= 1;
                c1_map_out5[i][j] <= 1;
                c1_map_out6[i][j] <= 1;
            end
        end
    end
    else if(mp_flag == 1 && store > -1 && shift1_row_cnt != 14)begin
        case(ch_cnt)
        4'd1:begin
            c1_map_out1[shift1_row_cnt][col_write_index] <= mp1_out;
            c1_map_out1[shift2_row_cnt][col_write_index] <= mp2_out;
            ch_cnt <= 1;
        end
        4'd2:begin
            c1_map_out2[shift1_row_cnt][col_write_index] <= mp1_out;
            c1_map_out2[shift2_row_cnt][col_write_index] <= mp2_out;
            ch_cnt <= 2;
        end
        4'd3:begin
            c1_map_out3[shift1_row_cnt][col_write_index] <= mp1_out;
            c1_map_out3[shift2_row_cnt][col_write_index] <= mp2_out;
            ch_cnt <= 3;
        end
        4'd4:begin
            c1_map_out4[shift1_row_cnt][col_write_index] <= mp1_out;
            c1_map_out4[shift2_row_cnt][col_write_index] <= mp2_out;
            ch_cnt <= 4;
        end
        4'd5:begin
            c1_map_out5[shift1_row_cnt][col_write_index] <= mp1_out;
            c1_map_out5[shift2_row_cnt][col_write_index] <= mp2_out;
            ch_cnt <= 5;
        end   
        4'd6:begin
            c1_map_out6[shift1_row_cnt][col_write_index] <= mp1_out;
            c1_map_out6[shift2_row_cnt][col_write_index] <= mp2_out;
            ch_cnt <= 6;
        end
        default:begin
            for (i=0; i<14; i=i+1)begin
                for(j=0; j<14; j=j+1)begin
                    c1_map_out1[i][j] <= 1;
                    c1_map_out2[i][j] <= 1;
                    c1_map_out3[i][j] <= 1;
                    c1_map_out4[i][j] <= 1;
                    c1_map_out5[i][j] <= 1;
                    c1_map_out6[i][j] <= 1;
                end
            end
        end
        endcase
    end
    else if(shift1_row_cnt == 14)begin
        ch_cnt <= ch_cnt + 1;
    end
end
//write
//0-13 col_write_index  0-15 sft1,sft2 counter
always@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        col_write_index <= 0;
        shift1_row_cnt <= 0;
        shift2_row_cnt <= 1;
    end
    else if (mp_flag == 1 && store > -1  && shift1_row_cnt != 14)begin
        if (col_write_index == 13)begin
            col_write_index <= 0;
            shift1_row_cnt <= shift1_row_cnt + 2;
            shift2_row_cnt <= shift2_row_cnt + 2;          
        end
        else begin
            col_write_index <= col_write_index + 1;
        end
    end
    else if(shift1_row_cnt == 14)begin
        shift1_row_cnt <= 0;
        shift2_row_cnt <= 1;
        col_write_index <= 0;
    end
end
//===================================================================//
//                      write                                        //
//===================================================================//
always @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        c1_out_flag_control <= 0;
    end
    else if (state == c1_write)begin
        case(c1_out_channel_cnt)
        4'd1:begin
            if(c1_out_flag_control == 1'b0) begin
                sram_act_wdata0 <= {c1_map_out1[3][out_row_index], c1_map_out1[2][out_row_index], c1_map_out1[1][out_row_index], c1_map_out1[0][out_row_index]};
                sram_act_wdata1 <= {c1_map_out1[7][out_row_index], c1_map_out1[6][out_row_index], c1_map_out1[5][out_row_index], c1_map_out1[4][out_row_index]};
                c1_out_flag_control <= 1;
            end 
            else begin
                sram_act_wdata0 <= {c1_map_out1[11][out_row_index], c1_map_out1[10][out_row_index], c1_map_out1[9][out_row_index], c1_map_out1[8][out_row_index]};
                sram_act_wdata1 <= {16'd0, c1_map_out1[13][out_row_index], c1_map_out1[12][out_row_index]};
                if(out_row_index == 13) begin
                    c1_out_flag_control <= 0;
                end 
                else begin
                    c1_out_flag_control <= 0;
                end
            end
        end
        4'd2:begin
            if(c1_out_flag_control == 1'b0) begin
                sram_act_wdata0 <= {c1_map_out2[3][out_row_index], c1_map_out2[2][out_row_index], c1_map_out2[1][out_row_index], c1_map_out2[0][out_row_index]};
                sram_act_wdata1 <= {c1_map_out2[7][out_row_index], c1_map_out2[6][out_row_index], c1_map_out2[5][out_row_index], c1_map_out2[4][out_row_index]};
                c1_out_flag_control <= 1;
            end 
            else begin
                sram_act_wdata0 <= {c1_map_out2[11][out_row_index], c1_map_out2[10][out_row_index], c1_map_out2[9][out_row_index], c1_map_out2[8][out_row_index]};
                sram_act_wdata1 <= {16'd0, c1_map_out2[13][out_row_index], c1_map_out2[12][out_row_index]};
                if(out_row_index == 13) begin
                    c1_out_flag_control <= 0;
                end 
                else begin
                    c1_out_flag_control <= 0;
                end
            end
        end
        4'd3:begin
            if(c1_out_flag_control == 1'b0) begin
                sram_act_wdata0 <= {c1_map_out3[3][out_row_index], c1_map_out3[2][out_row_index], c1_map_out3[1][out_row_index], c1_map_out3[0][out_row_index]};
                sram_act_wdata1 <= {c1_map_out3[7][out_row_index], c1_map_out3[6][out_row_index], c1_map_out3[5][out_row_index], c1_map_out3[4][out_row_index]};
                c1_out_flag_control <= 1;
            end 
            else begin
                sram_act_wdata0 <= {c1_map_out3[11][out_row_index], c1_map_out3[10][out_row_index], c1_map_out3[9][out_row_index], c1_map_out3[8][out_row_index]};
                sram_act_wdata1 <= {16'd0, c1_map_out3[13][out_row_index], c1_map_out3[12][out_row_index]};
                if(out_row_index == 13) begin
                    c1_out_flag_control <= 0;
                end 
                else begin
                    c1_out_flag_control <= 0;
                end
            end
        end
        4'd4:begin
            if(c1_out_flag_control == 1'b0) begin
                sram_act_wdata0 <= {c1_map_out4[3][out_row_index], c1_map_out4[2][out_row_index], c1_map_out4[1][out_row_index], c1_map_out4[0][out_row_index]};
                sram_act_wdata1 <= {c1_map_out4[7][out_row_index], c1_map_out4[6][out_row_index], c1_map_out4[5][out_row_index], c1_map_out4[4][out_row_index]};
                c1_out_flag_control <= 1;
            end 
            else begin
                sram_act_wdata0 <= {c1_map_out4[11][out_row_index], c1_map_out4[10][out_row_index], c1_map_out4[9][out_row_index], c1_map_out4[8][out_row_index]};
                sram_act_wdata1 <= {16'd0, c1_map_out4[13][out_row_index], c1_map_out4[12][out_row_index]};
                if(out_row_index == 13) begin
                    c1_out_flag_control <= 0;
                end 
                else begin
                    c1_out_flag_control <= 0;
                end
            end
        end
        4'd5:begin
            if(c1_out_flag_control == 1'b0) begin
                sram_act_wdata0 <= {c1_map_out5[3][out_row_index], c1_map_out5[2][out_row_index], c1_map_out5[1][out_row_index], c1_map_out5[0][out_row_index]};
                sram_act_wdata1 <= {c1_map_out5[7][out_row_index], c1_map_out5[6][out_row_index], c1_map_out5[5][out_row_index], c1_map_out5[4][out_row_index]};
                c1_out_flag_control <= 1;
            end 
            else begin
                sram_act_wdata0 <= {c1_map_out5[11][out_row_index], c1_map_out5[10][out_row_index], c1_map_out5[9][out_row_index], c1_map_out5[8][out_row_index]};
                sram_act_wdata1 <= {16'd0, c1_map_out5[13][out_row_index], c1_map_out5[12][out_row_index]};
                if(out_row_index == 13) begin
                    c1_out_flag_control <= 0;
                end 
                else begin
                    c1_out_flag_control <= 0;
                end
            end
        end
        4'd6:begin
            if(c1_out_flag_control == 1'b0) begin
                sram_act_wdata0 <= {c1_map_out6[3][out_row_index], c1_map_out6[2][out_row_index], c1_map_out6[1][out_row_index], c1_map_out6[0][out_row_index]};
                sram_act_wdata1 <= {c1_map_out6[7][out_row_index], c1_map_out6[6][out_row_index], c1_map_out6[5][out_row_index], c1_map_out6[4][out_row_index]};
                c1_out_flag_control <= 1;
            end 
            else begin
                sram_act_wdata0 <= {c1_map_out6[11][out_row_index], c1_map_out6[10][out_row_index], c1_map_out6[9][out_row_index], c1_map_out6[8][out_row_index]};
                sram_act_wdata1 <= {16'd0, c1_map_out6[13][out_row_index], c1_map_out6[12][out_row_index]};
                if(out_row_index == 13) begin
                    c1_out_flag_control <= 0;
                end 
                else begin
                    c1_out_flag_control <= 0;
                end
            end
        end
        default:begin
            c1_out_flag_control <= 0;
        end
        endcase
    end
    else if (state == c2_write)begin
        case(c2_write_cycle_cnt)
        4'd0:begin
            case(c2_write_index_cnt)
            4'd0:begin
                sram_act_wdata0 <= {c2_mp_out[3][0], c2_mp_out[2][0], c2_mp_out[1][0], c2_mp_out[0][0]};
                sram_act_wdata1 <= {c2_mp_out[2][1], c2_mp_out[1][1], c2_mp_out[0][1], c2_mp_out[4][0]};
            end
            4'd1:begin
                sram_act_wdata0 <= {c2_mp_out[1][2], c2_mp_out[0][2], c2_mp_out[4][1], c2_mp_out[3][1]};
                sram_act_wdata1 <= {c2_mp_out[0][3], c2_mp_out[4][2], c2_mp_out[3][2], c2_mp_out[2][2]};
            end
            4'd2:begin
                sram_act_wdata0 <= {c2_mp_out[4][3], c2_mp_out[3][3], c2_mp_out[2][3], c2_mp_out[1][3]};
                sram_act_wdata1 <= {c2_mp_out[3][4], c2_mp_out[2][4], c2_mp_out[1][4], c2_mp_out[0][4]};
            end
            4'd3:begin
                sram_act_wdata0 <= {24'd0, c2_mp_out[4][4]};
                sram_act_wdata1 <= {32'd0};
            end
            default:begin
                sram_act_wdata0 <= 1;
                sram_act_wdata1 <= 1;
            end
            endcase
        end
        4'd1:begin
            case(c2_write_index_cnt)
            4'd0:begin
                sram_act_wdata0 <= {c2_mp_out[2][0], c2_mp_out[1][0], c2_mp_out[0][0], 8'd0};
                sram_act_wdata1 <= {c2_mp_out[1][1], c2_mp_out[0][1], c2_mp_out[4][0], c2_mp_out[3][0]};
            end
            4'd1:begin
                sram_act_wdata0 <= {c2_mp_out[0][2], c2_mp_out[4][1], c2_mp_out[3][1], c2_mp_out[2][1]};
                sram_act_wdata1 <= {c2_mp_out[4][2], c2_mp_out[3][2], c2_mp_out[2][2], c2_mp_out[1][2]};
            end
            4'd2:begin
                sram_act_wdata0 <= {c2_mp_out[3][3], c2_mp_out[2][3], c2_mp_out[1][3], c2_mp_out[0][3]};
                sram_act_wdata1 <= {c2_mp_out[2][4], c2_mp_out[1][4], c2_mp_out[0][4], c2_mp_out[4][3]};
            end
            4'd3:begin
                sram_act_wdata0 <= {16'd0, c2_mp_out[4][4], c2_mp_out[3][4]};
                sram_act_wdata1 <= {32'd0};
            end
            default:begin
                sram_act_wdata0 <= 1;
                sram_act_wdata1 <= 1;
            end
            endcase
        end
        4'd2:begin
            case(c2_write_index_cnt)
            4'd0:begin
                sram_act_wdata0 <= {c2_mp_out[1][0], c2_mp_out[0][0], 16'd0};
                sram_act_wdata1 <= {c2_mp_out[0][1], c2_mp_out[4][0], c2_mp_out[3][0], c2_mp_out[2][0]};
            end
            4'd1:begin
                sram_act_wdata0 <= {c2_mp_out[4][1], c2_mp_out[3][1], c2_mp_out[2][1], c2_mp_out[1][1]};
                sram_act_wdata1 <= {c2_mp_out[3][2], c2_mp_out[2][2], c2_mp_out[1][2], c2_mp_out[0][2]};
            end
            4'd2:begin
                sram_act_wdata0 <= {c2_mp_out[2][3], c2_mp_out[1][3], c2_mp_out[0][3], c2_mp_out[4][2]};
                sram_act_wdata1 <= {c2_mp_out[1][4], c2_mp_out[0][4], c2_mp_out[4][3], c2_mp_out[3][3]};
            end
            4'd3:begin
                sram_act_wdata0 <= {8'd0, c2_mp_out[4][4], c2_mp_out[3][4], c2_mp_out[2][4]};
                sram_act_wdata1 <= {32'd0};
            end
            default:begin
                sram_act_wdata0 <= 1;
                sram_act_wdata1 <= 1;
            end
            endcase
        end
        4'd3:begin
            case(c2_write_index_cnt)
            4'd0:begin
                sram_act_wdata0 <= {c2_mp_out[0][0], 24'd0};
                sram_act_wdata1 <= {c2_mp_out[4][0], c2_mp_out[3][0], c2_mp_out[2][0], c2_mp_out[1][0]};
            end
            4'd1:begin
                sram_act_wdata0 <= {c2_mp_out[3][1], c2_mp_out[2][1], c2_mp_out[1][1], c2_mp_out[0][1]};
                sram_act_wdata1 <= {c2_mp_out[2][2], c2_mp_out[1][2], c2_mp_out[0][2], c2_mp_out[4][1]};
            end
            4'd2:begin
                sram_act_wdata0 <= {c2_mp_out[1][3], c2_mp_out[0][3], c2_mp_out[4][2], c2_mp_out[3][2]};
                sram_act_wdata1 <= {c2_mp_out[0][4], c2_mp_out[4][3], c2_mp_out[3][3], c2_mp_out[2][3]};
            end
            4'd3:begin
                sram_act_wdata0 <= {c2_mp_out[4][4], c2_mp_out[3][4], c2_mp_out[2][4], c2_mp_out[1][4]};
                sram_act_wdata1 <= {32'd0};
            end
            default:begin
                sram_act_wdata0 <= 1;
                sram_act_wdata1 <= 1;
            end
            endcase
        end 
        default:begin
            sram_act_wdata0 <= 1;
            sram_act_wdata1 <= 1;
        end
        endcase
    end
    else if (state == c3_write)begin
        sram_act_wdata0 <= {c3_fc_buffer[cnt_c3_out_index+4'd3],c3_fc_buffer[cnt_c3_out_index+4'd2],c3_fc_buffer[cnt_c3_out_index+4'd1],c3_fc_buffer[cnt_c3_out_index]};
        sram_act_wdata1 <= {c3_fc_buffer[cnt_c3_out_index+4'd7],c3_fc_buffer[cnt_c3_out_index+4'd6],c3_fc_buffer[cnt_c3_out_index+4'd5],c3_fc_buffer[cnt_c3_out_index+4'd4]};
    end
    else if (state ==fc1_write)begin
        if (fc1_out_index == 7'd80)begin
            sram_act_wdata0 <= {fc1_out_act[fc1_out_index+3], fc1_out_act[fc1_out_index+2], fc1_out_act[fc1_out_index+1], fc1_out_act[fc1_out_index]};
        end
        else if (fc1_out_index != 7'd80)begin
            sram_act_wdata0 <= {fc1_out_act[fc1_out_index+3], fc1_out_act[fc1_out_index+2], fc1_out_act[fc1_out_index+1], fc1_out_act[fc1_out_index]};
            sram_act_wdata1 <= {fc1_out_act[fc1_out_index+7], fc1_out_act[fc1_out_index+6], fc1_out_act[fc1_out_index+5], fc1_out_act[fc1_out_index+4]};
        end
        else begin
            sram_act_wdata0 <= 1;
            sram_act_wdata1 <= 1;
        end
    end
    else if (state == fc2_write)begin
        sram_act_wdata0 <= fc2_out[fc2_out_index];
        sram_act_wdata1 <= fc2_out[fc2_out_index + 1];
    end
    else begin
        sram_act_wdata0 <= 1;
        sram_act_wdata1 <= 1;
    end
end
//===================================================================//
//                     c1_write counter                              //
//===================================================================//
always @(posedge clk or negedge rst_n) begin
    if (~rst_n)begin
        c1_out_channel_cnt <= 1;
        out_row_index <= 0;
    end
    else if (state == c1_write)begin
        case(c1_out_channel_cnt)
        4'd1,4'd2,4'd3,4'd4,4'd5,4'd6:begin
            if(c1_out_flag_control == 1'b0) begin
                out_row_index <= out_row_index;
            end 
            else begin
                if(out_row_index == 13) begin
                    out_row_index <= 0;
                    c1_out_channel_cnt <= c1_out_channel_cnt + 1;
                end 
                else begin
                    out_row_index <= out_row_index + 1;
                    c1_out_channel_cnt <= c1_out_channel_cnt;
                end
            end
        end
        default:begin
            out_row_index <=0;
            //c1_out_flag_control <= 0;
        end
        endcase
    end
end
//===================================================================//
//                      c2                                           //
//===================================================================//
//load weight_map (5*5)
always @(posedge clk or negedge rst_n) begin
    if(~rst_n)
      read_c2_weight <= 0;
    else 
      read_c2_weight <= (state == r_c2_w) ? 1:0; 
end
always@(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        c2_weight_index <= 4'd0;
        for (i=0; i<5; i=i+1) begin
            for (j=0; j<5; j=j+1)begin
                c2_weight_map[i][j] <= 0;
            end
        end
    end
    //use for holding weights address
    else if(state != r_c2_w) begin
        c2_weight_index <= 4'd0;
        for (i=0; i<5; i=i+1) begin
            for (j=0; j<5; j=j+1)begin
                c2_weight_map[i][j] <= c2_weight_map[i][j];
            end
        end
    end
    else begin
        if(read_c2_weight == 1)begin
            c2_weight_index <= c2_weight_index + 4'd1;
            c2_weight_map[0][c2_weight_index] <= sram_weight_rdata0[7:0];
            c2_weight_map[1][c2_weight_index] <= sram_weight_rdata0[15:8];
            c2_weight_map[2][c2_weight_index] <= sram_weight_rdata0[23:16];
            c2_weight_map[3][c2_weight_index] <= sram_weight_rdata0[31:24];
            c2_weight_map[4][c2_weight_index] <= sram_weight_rdata1[7:0];
        end
    end
end
//load  c1_ifmap
always@(posedge clk or negedge rst_n) begin
    if(~rst_n) begin
        for (i=0; i<8; i=i+1) begin
            for (j=0; j<5; j=j+1) 
                c2_ifmap [i][j] <= 0;
        end
    end
    else if (state == c2_exe) begin
        c2_ifmap[0][4] <= sram_act_rdata0[7:0];
        c2_ifmap[1][4] <= sram_act_rdata0[15:8];
        c2_ifmap[2][4] <= sram_act_rdata0[23:16];
        c2_ifmap[3][4] <= sram_act_rdata0[31:24];
        c2_ifmap[4][4] <= sram_act_rdata1[7:0];
        c2_ifmap[5][4] <= sram_act_rdata1[15:8];
        c2_ifmap[6][4] <= sram_act_rdata1[23:16];
        c2_ifmap[7][4] <= sram_act_rdata1[31:24];
        for (i=0; i<8 ; i=i+1) begin
            for (j=0; j<4; j=j+1) begin
                c2_ifmap[i][j] <= c2_ifmap[i][j+1];
            end
        end
    end             
end
//c2_cal_cycle_cnt 0-13完成load activation
always@(posedge clk) begin
    if(state == c2_exe && c2_cal_cycle_cnt < 14 )begin
        c2_cal_cycle_cnt <= c2_cal_cycle_cnt + 1;
    end
    else if (c2_cal_cycle_cnt == 14)begin
        c2_cal_cycle_cnt <= 1;    
    end
    else begin
        c2_cal_cycle_cnt <= -1;
    end
end
assign c2_cal = (c2_cal_cycle_cnt > 4) ? 1 : 0; //if_map load up
//mac
always @* begin
    if(c2_cal)begin
        for (i=0; i<5; i=i+1)begin
            for (j=0; j<5; j=j+1)begin
                p1_c2[i][j] = c2_weight_map[i][j] * c2_ifmap[i][j];
                p2_c2[i][j] = c2_weight_map[i][j] * c2_ifmap[i+1][j];
                p3_c2[i][j] = c2_weight_map[i][j] * c2_ifmap[i+2][j];
                p4_c2[i][j] = c2_weight_map[i][j] * c2_ifmap[i+3][j];
            end
        end
    end
    else begin
        for (i=0; i<5; i=i+1)begin
            for (j=0; j<5; j=j+1)begin
                p1_c2[i][j] = 0;
                p2_c2[i][j] = 0;
                p3_c2[i][j] = 0;
                p4_c2[i][j] = 0;
            end
        end
    end
end
//------------------------------------------------------------//
always @* begin
    if(c2_cal)begin
        c2_psum1 = 0;
        c2_psum2 = 0;
        c2_psum3 = 0;
        c2_psum4 = 0;
        for (i=0; i<5;i=i+1) begin
            for (j=0; j<5; j=j+1) begin
                c2_psum1 = c2_psum1 + p1_c2[i][j];
                c2_psum2 = c2_psum2 + p2_c2[i][j];
                c2_psum3 = c2_psum3 + p3_c2[i][j];
                c2_psum4 = c2_psum4 + p4_c2[i][j];
            end
        end
    end
    else begin
        c2_psum1 = 0;
        c2_psum2 = 0;
        c2_psum3 = 0;
        c2_psum4 = 0;
    end
end
//------------------------------------------------------------//
always @(posedge clk  or negedge rst_n)begin
    if (~rst_n)begin
        for(i=0; i<10; i=i+1)begin
            for(j=0; j<10; j=j+1)begin
                c2_conv_temp[i][j] <= 1;
            end
        end
    end
    else if (c2_cal)begin
        case(c2_cal_cycle_control)
        4'd1:begin
            c2_conv_temp[0][pre_mp_index] <= c2_psum1;
            c2_conv_temp[1][pre_mp_index] <= c2_psum2;
            c2_conv_temp[2][pre_mp_index] <= c2_psum3;
            c2_conv_temp[3][pre_mp_index] <= c2_psum4;
        end
        4'd2:begin
            c2_conv_temp[4][pre_mp_index] <= c2_psum1;
            c2_conv_temp[5][pre_mp_index] <= c2_psum2;
            c2_conv_temp[6][pre_mp_index] <= c2_psum3;
            c2_conv_temp[7][pre_mp_index] <= c2_psum4;
        end
        4'd3:begin
            c2_conv_temp[8][pre_mp_index] <= c2_psum1;
            c2_conv_temp[9][pre_mp_index] <= c2_psum2;
        end
        default:begin
            for(i=0; i<10; i=i+1)begin
                for(j=0; j<10; j=j+1)begin
                    c2_conv_temp[i][j] <= 1;
                end
            end
        end
        endcase
    end
    else if (c2_cal_cycle_control == 4'd4)begin
        for(i=0; i<10; i=i+1)begin
            for(j=0; j<10; j=j+1)begin
                c2_conv_temp[i][j] <= 1;
            end
        end
    end
end
//------------------------------------------------------------//
always @(posedge clk  or negedge rst_n)begin
    if (~rst_n)begin
        pre_mp_index <= 4'd0;
        c2_cal_cycle_control <= 4'd1;
    end
    else if (c2_cal && c2_cal_cycle_control < 4'd4)begin
        if (pre_mp_index == 4'd9)begin
            pre_mp_index <= 0;
            c2_cal_cycle_control <= c2_cal_cycle_control + 4'd1;   
        end
        else begin
            pre_mp_index <= pre_mp_index + 4'd1;
        end
    end
    else if (c2_cal_cycle_control == 4'd4)begin
        pre_mp_index <= pre_mp_index ;
        c2_cal_cycle_control <= 4'd1;
    end
end
//存值+累加
always @(posedge clk or negedge rst_n)begin
    if(~rst_n)begin
        for(i=0; i<10; i=i+1)begin
            for(j=0; j<10; j=j+1)begin
                c2_temp_sum[i][j] <= 0;
            end
        end
    end
    else if (c2_cal_cycle_control == 4'd4)begin
        for(i=0; i<10; i=i+1)begin
            for(j=0; j<10; j=j+1)begin
                c2_temp_sum[i][j] <= c2_temp_sum[i][j] + c2_conv_temp[i][j];
            end
        end
    end
    else if(state == c2_write && c2_write_index_cnt == 4'd4)begin
        for(i=0; i<10; i=i+1)begin
            for(j=0; j<10; j=j+1)begin
                c2_temp_sum[i][j] <= 0;
            end
        end
    end
    else begin
        for(i=0; i<10; i=i+1)begin
            for(j=0; j<10; j=j+1)begin
                c2_temp_sum[i][j] <= c2_temp_sum[i][j];
            end
        end
    end
end
//------------------------------------------------------------//
always @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        c2_ifmap_channel_cnt <= 4'd0;
    end
    else begin
        c2_ifmap_channel_cnt <= c2_ifmap_channel_cnt_tmp;
    end
end
//------------------------------------------------------------//
always @*begin
    if (~rst_n)begin
        c2_ifmap_channel_cnt_tmp = 4'd0;
    end
    else if (c2_cal_cycle_control == 4'd4)begin
        c2_ifmap_channel_cnt_tmp = c2_ifmap_channel_cnt + 4'd1;
    end
    else if(state == c2_write && c2_write_index_cnt == 4'd4)begin
        c2_ifmap_channel_cnt_tmp =  c2_ifmap_channel_cnt;
    end
    else begin
        if(c2_ifmap_channel_cnt == 4'd6)begin
             c2_ifmap_channel_cnt_tmp = 4'd0;  
        end
        else begin
            c2_ifmap_channel_cnt_tmp = c2_ifmap_channel_cnt;
        end
    end
end
//re_quantize
always @* begin
    case(state)
        c2_write:begin
            for(i=0; i<10; i=i+1)begin
                for(j=0; j<10; j=j+1)begin
                    c2_temp_relu[i][j] = (c2_temp_sum[i][j] > 0 )? c2_temp_sum[i][j]:0;
                    c2_temp_scale[i][j] = c2_temp_relu[i][j] * scale_CONV2;
                    c2_temp_shift[i][j] = c2_temp_scale[i][j] >> 16;
                    c2_temp_clamp[i][j] = (c2_temp_shift[i][j] >127)? 127 : c2_temp_shift[i][j];
                end
            end
        end
        default: begin
            for(i=0; i<10; i=i+1)begin
                for(j=0; j<10; j=j+1)begin
                    c2_temp_relu[i][j] = 0;
                    c2_temp_scale[i][j] = 0;
                    c2_temp_shift[i][j] = 0;
                    c2_temp_clamp[i][j] = 0;
                end
            end
        end
    endcase
end
//conv2_maxpooling
always @* begin
    case(state)
        c2_write:begin
            for(i=0; i<5; i=i+1)begin
                for(j=0; j<5; j=j+1)begin
                    c2_mp_out[i][j] = c2_temp_clamp[i*2][j*2] >= c2_temp_clamp[i*2+1][j*2] 
                    && c2_temp_clamp[i*2][j*2] >= c2_temp_clamp[i*2][j*2+1] 
                    && c2_temp_clamp[i*2][j*2] >= c2_temp_clamp[i*2+1][j*2+1] ? c2_temp_clamp[i*2][j*2] : 
                    (c2_temp_clamp[i*2+1][j*2] >= c2_temp_clamp[i*2][j*2+1] && c2_temp_clamp[i*2+1][j*2] 
                    >= c2_temp_clamp[i*2+1][j*2+1] ? c2_temp_clamp[i*2+1][j*2] : (c2_temp_clamp[i*2][j*2+1] 
                    >= c2_temp_clamp[i*2+1][j*2+1] ? c2_temp_clamp[i*2][j*2+1] : c2_temp_clamp[i*2+1][j*2+1]));
                end
            end
        end
        default:begin
            for(i=0; i<5; i=i+1)begin
                for(j=0; j<5; j=j+1)begin
                c2_mp_out[i][j] = 0;
                end
            end
        end      
    endcase
end

//control counter
always @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        c2_write_index_cnt <= 4'd0;
        cnt_c2_out_channel <= 4'd0;
        c2_write_cycle_cnt <= 4'd0;
    end
    else if (state == c2_write)begin
        case(c2_write_cycle_cnt)
        4'd0,4'd1,4'd2:begin
            case(c2_write_index_cnt)
            4'd0 , 4'd1 , 4'd2 , 4'd3 :begin
                c2_write_index_cnt <= c2_write_index_cnt + 4'd1;
                cnt_c2_out_channel <= cnt_c2_out_channel;
                c2_write_cycle_cnt <= c2_write_cycle_cnt;
            end
            4'd4:begin
                c2_write_index_cnt <= 0;
                cnt_c2_out_channel <= cnt_c2_out_channel + 4'd1;
                c2_write_cycle_cnt <= c2_write_cycle_cnt + 4'd1;
            end
            default:begin
                c2_write_index_cnt <= 0;
                cnt_c2_out_channel <= 4'd0;
                c2_write_cycle_cnt <= 4'd0;
            end
            endcase
        end
        4'd3:begin
            case(c2_write_index_cnt)
            4'd0 , 4'd1 , 4'd2 , 4'd3 :begin
                c2_write_index_cnt <= c2_write_index_cnt + 4'd1;
                cnt_c2_out_channel <= cnt_c2_out_channel;
                c2_write_cycle_cnt <= c2_write_cycle_cnt;
            end
            4'd4:begin
                c2_write_index_cnt <= 0;
                cnt_c2_out_channel <= cnt_c2_out_channel + 4'd1;
                c2_write_cycle_cnt <= 4'd0;
            end
            default:begin
                c2_write_index_cnt <= 0;
                cnt_c2_out_channel <= 4'd0;
                c2_write_cycle_cnt <= 4'd0;
            end
            endcase
        end 
        default:begin
            c2_write_index_cnt <= 4'd0;
            cnt_c2_out_channel <= 4'd0;
            c2_write_cycle_cnt <= 4'd0;
        end
        endcase
    end
end
//c3 calculate
assign c3_cal = (c3_exe_cnt > -1) ? 1 : 0;
    //exe_cnt
always @(posedge clk or negedge rst_n) begin
    if (~rst_n)begin
        c3_col_index <= 0; //0-120
    end
    else begin
        c3_col_index <= c3_col_index_tmp;
    end
end
//------------------------------------------------------------//
always @(posedge clk or negedge rst_n) begin
    if (~rst_n)begin
        c3_exe_cnt <= -1; //0-49
    end
    else begin
        c3_exe_cnt <= c3_exe_cnt_tmp;
    end
end
//------------------------------------------------------------//
always @*begin 
    if (state == c3_exe && c3_exe_cnt != 49)begin
            c3_exe_cnt_tmp = c3_exe_cnt + 1;
            c3_col_index_tmp = c3_col_index;
    end
    else if (state == c3_exe && c3_exe_cnt == 49)begin
            c3_exe_cnt_tmp = 0;
            c3_col_index_tmp = c3_col_index + 1;
    end
    else begin
        c3_col_index_tmp = 0;
        c3_exe_cnt_tmp = -1;
    end                    
end
//------------------------------------------------------------//
always @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        c3_psum <= 0;
        for(i=0;i<120;i=i+1)begin
            c3_fc_buffer[i] <= 0;
        end    
    end
    else if (state == c3_exe && c3_exe_cnt != 49)begin
        c3_psum <= c3_psum + c3_mac; 
    end
    else if (c3_exe_cnt == 49)begin
        c3_psum <=0;
        c3_fc_buffer[c3_col_index] <= c3_fc_clamp[7:0];
    end
end
//------------------------------------------------------------//
always @* begin
    if (c3_cal)begin
        c3_mac = {{24{sram_act_rdata0[7]}}, sram_act_rdata0[7:0]} * {{24{sram_weight_rdata0[7]}}, sram_weight_rdata0[7:0]}
               + {{24{sram_act_rdata0[15]}}, sram_act_rdata0[15:8]} * {{24{sram_weight_rdata0[15]}}, sram_weight_rdata0[15:8]}
               + {{24{sram_act_rdata0[23]}}, sram_act_rdata0[23:16]} * {{24{sram_weight_rdata0[23]}}, sram_weight_rdata0[23:16]}
               + {{24{sram_act_rdata0[31]}}, sram_act_rdata0[31:24]} * {{24{sram_weight_rdata0[31]}}, sram_weight_rdata0[31:24]}
               + {{24{sram_act_rdata1[7]}}, sram_act_rdata1[7:0]} * {{24{sram_weight_rdata1[7]}}, sram_weight_rdata1[7:0]}
               + {{24{sram_act_rdata1[15]}}, sram_act_rdata1[15:8]} * {{24{sram_weight_rdata1[15]}}, sram_weight_rdata1[15:8]}
               + {{24{sram_act_rdata1[23]}}, sram_act_rdata1[23:16]} * {{24{sram_weight_rdata1[23]}}, sram_weight_rdata1[23:16]} 
               + {{24{sram_act_rdata1[31]}}, sram_act_rdata1[31:24]} * {{24{sram_weight_rdata1[31]}}, sram_weight_rdata1[31:24]};
    end
    else begin
        c3_mac = 0;
    end
end
//------------------------------------------------------------//
always @* begin
    if (c3_cal)begin
        c3_sum = c3_psum + c3_mac;
        c3_fc_relu = (c3_sum[31] == 1) ? 0 : c3_sum;
        c3_fc_scale = c3_fc_relu * scale_CONV3;
        c3_fc_shift = c3_fc_scale >> 16;
        c3_fc_clamp = (c3_fc_shift > 127) ? 127 : c3_fc_shift;
    end
    else begin
        c3_sum = 0;
        c3_fc_relu = 0;
        c3_fc_scale = 0;
        c3_fc_shift = 0;
        c3_fc_clamp = 0;
    end   
end
//write conv3
always@(posedge clk or negedge rst_n)begin
    if (~rst_n)begin
        cnt_c3_out_index <= 4'd0;
    end
    else if (state == c3_write)begin
        cnt_c3_out_index <= cnt_c3_out_index_tmp;
    end
    else begin
        cnt_c3_out_index <= 4'd0;
    end
end
//cnt_c3_out_index for write c3 output
always @*begin
    case(state)
        c3_write:begin
            cnt_c3_out_index_tmp = cnt_c3_out_index + 4'd8;
        end
        default:begin
            cnt_c3_out_index_tmp = 4'd0;
        end
    endcase
end
//------------------------------------fc1-----------------------------------------------//
always @(posedge clk or negedge rst_n) begin
    if (~rst_n)begin
        fc1_index <= 0; //84
    end
    else begin
        fc1_index <= fc1_index_tmp;
    end
end
//exe_cnt
always @(posedge clk or negedge rst_n) begin
    if (~rst_n)begin
         exe_cnt <= -1; 
    end
    else begin
        exe_cnt <= exe_cnt_next;
    end
end
//------------------------------------------------------------//
always @*begin 
    if(state == fc1_exe && exe_cnt != 14)begin
        exe_cnt_next = exe_cnt + 1;
        fc1_index_tmp = fc1_index;
    end
    else if (state == fc1_exe && exe_cnt == 14)begin
        exe_cnt_next = 0;
        fc1_index_tmp = fc1_index + 1;
    end
    else begin
        fc1_index_tmp = 0;
        exe_cnt_next = -1;
    end                    
end
//------------------------------------------------------------//
assign fc1_cal = (exe_cnt  > -1)?1:0;
//------------------------------------------------------------//
always @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        fc1_p_sum <= 0;
        for(i=0;i<84;i=i+1)begin
            fc1_out_act[i] <= 0;
        end    
    end
    else if (state == fc1_exe && exe_cnt != 14)begin
        fc1_p_sum <= fc1_p_sum + fc1_mac; 
    end
    else if (exe_cnt == 14)begin
        fc1_p_sum <=0;
        fc1_out_act[fc1_index] <= clip_fc1[7:0];
    end
end
//calculation
always @* begin
    if (fc1_cal)begin
        fc1_mac =  {{24{sram_act_rdata0[31]}},sram_act_rdata0[31:24]} * {{24{sram_weight_rdata0[31]}},sram_weight_rdata0[31:24]}
                    + {{24{sram_act_rdata0[23]}},sram_act_rdata0[23:16]} * {{24{sram_weight_rdata0[23]}},sram_weight_rdata0[23:16]}
                    + {{24{sram_act_rdata0[15]}},sram_act_rdata0[15:8]} * {{24{sram_weight_rdata0[15]}},sram_weight_rdata0[15:8]}
                    + {{24{sram_act_rdata0[7]}},sram_act_rdata0[7:0]} * {{24{sram_weight_rdata0[7]}},sram_weight_rdata0[7:0]}
                    + {{24{sram_act_rdata1[31]}},sram_act_rdata1[31:24]} * {{24{sram_weight_rdata1[31]}},sram_weight_rdata1[31:24]}
                    + {{24{sram_act_rdata1[23]}},sram_act_rdata1[23:16]} * {{24{sram_weight_rdata1[23]}},sram_weight_rdata1[23:16]}
                    + {{24{sram_act_rdata1[15]}},sram_act_rdata1[15:8]} * {{24{sram_weight_rdata1[15]}},sram_weight_rdata1[15:8]}
                    + {{24{sram_act_rdata1[7]}},sram_act_rdata1[7:0]} * {{24{sram_weight_rdata1[7]}},sram_weight_rdata1[7:0]};
    end
    else begin
        fc1_mac =0;
    end
end
//------------------------------------------------------------//
always @* begin
    if (fc1_cal)begin
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
//fc1_out_index
always @(posedge clk or negedge rst_n) begin
    if (~rst_n)begin
        fc1_out_index <= 0;    
    end
    else begin
        fc1_out_index <=fc1_out_index_tmp;
    end
end
//------------------------------------------------------------//
always @*begin
    if (state == fc1_write )begin
        fc1_out_index_tmp = fc1_out_index + 8;  
    end
    else begin
        fc1_out_index_tmp = 0;  
    end
end

//load 10個weights bias
always @(posedge clk or negedge rst_n)begin
    if (~rst_n)begin
        for (i=0;i<10;i=i+1)begin
            fc2_bias_weight_buffer[i] <= 0;
        end
        fc2_bias_weight_index <= 0;
    end
    else if (state == fc2_bias && fc2_bias_weight_index != 10)begin
        fc2_bias_weight_buffer[fc2_bias_weight_index]  <= sram_weight_rdata0;
        fc2_bias_weight_buffer[fc2_bias_weight_index+1]<= sram_weight_rdata1;
        fc2_bias_weight_index <= fc2_bias_weight_index_tmp;
    end
end
//------------------------------------------------------------//
always @*begin
    if (state == fc2_bias && fc2_bias_weight_index != 10)begin
        fc2_bias_weight_index_tmp = fc2_bias_weight_index +2;
    end
    else begin
        fc2_bias_weight_index_tmp = 0;
    end
end
//fc2_index counter
always @(posedge clk or negedge rst_n) begin
    if (~rst_n)begin
        fc2_index <= 0; //84
    end
    else begin
        fc2_index <= fc2_index_tmp;
    end
end
//fc2_exe counter
always @(posedge clk or negedge rst_n) begin
    if (~rst_n)begin
        fc2_exe_cnt <= -1; 
    end
    else begin
        fc2_exe_cnt <= fc2_exe_cnt_tmp;
    end
end
//combinational logic of fc2_exe & fc2_index counter
always @*begin 
    if(state == fc2_exe && fc2_exe_cnt != 10)begin
        fc2_exe_cnt_tmp = fc2_exe_cnt + 1;
        fc2_index_tmp = fc2_index;
    end
    else if (state == fc2_exe && fc2_exe_cnt == 10)begin
        fc2_exe_cnt_tmp = 0;
        fc2_index_tmp = fc2_index + 1;
    end
    else begin
        fc2_index_tmp = 0;
        fc2_exe_cnt_tmp = -1;
    end                    
end
///////////////////////////////////////////////////////
assign fc2_cal = (fc2_exe_cnt > -1)?1:0;
/////////////////////////////////////////////// 
always @(posedge clk or negedge rst_n) begin
    if(~rst_n)begin
        fc2_psum <= 0;
        fc2_add_bias_done <=0;
        for(i=0;i<10;i=i+1)begin
            fc2_out[fc2_index] <= 0;
        end    
    end
    else if (state == fc2_exe && fc2_exe_cnt != 10)begin
        fc2_psum <= fc2_psum + fc2_mac; 
    end
    else if (state == fc2_exe && fc2_exe_cnt == 10)begin
        fc2_psum <=0;
        fc2_out[fc2_index] <= fc2_sum;
    end
    else if (state == fc2_bias && fc2_bias_weight_index == 10 && fc2_add_bias_done == 0)begin
        for(i=0; i<10; i=i+1)begin
            fc2_out[i] <= fc2_out[i] + fc2_bias_weight_buffer[i];            
        end
        fc2_add_bias_done <=1;
    end
end
//------------------------------------------------------------//
always @* begin
    if (fc2_cal)begin
        //+bias
        if (fc2_exe_cnt == 10)begin
            fc2_mac = {{24{sram_act_rdata0[7]}}, sram_act_rdata0[7:0]} * {{24{sram_weight_rdata0[7]}}, sram_weight_rdata0[7:0]}
                    + {{24{sram_act_rdata0[15]}}, sram_act_rdata0[15:8]} * {{24{sram_weight_rdata0[15]}}, sram_weight_rdata0[15:8]}
                    + {{24{sram_act_rdata0[23]}}, sram_act_rdata0[23:16]} * {{24{sram_weight_rdata0[23]}}, sram_weight_rdata0[23:16]}
                    + {{24{sram_act_rdata0[31]}}, sram_act_rdata0[31:24]} * {{24{sram_weight_rdata0[31]}}, sram_weight_rdata0[31:24]};            
        end
        else begin
            fc2_mac = {{24{sram_act_rdata0[7]}}, sram_act_rdata0[7:0]} * {{24{sram_weight_rdata0[7]}}, sram_weight_rdata0[7:0]}
                    + {{24{sram_act_rdata0[15]}}, sram_act_rdata0[15:8]} * {{24{sram_weight_rdata0[15]}}, sram_weight_rdata0[15:8]}
                    + {{24{sram_act_rdata0[23]}}, sram_act_rdata0[23:16]} * {{24{sram_weight_rdata0[23]}}, sram_weight_rdata0[23:16]}
                    + {{24{sram_act_rdata0[31]}}, sram_act_rdata0[31:24]} * {{24{sram_weight_rdata0[31]}}, sram_weight_rdata0[31:24]}
                    + {{24{sram_act_rdata1[7]}}, sram_act_rdata1[7:0]} * {{24{sram_weight_rdata1[7]}}, sram_weight_rdata1[7:0]}
                    + {{24{sram_act_rdata1[15]}}, sram_act_rdata1[15:8]} * {{24{sram_weight_rdata1[15]}}, sram_weight_rdata1[15:8]}
                    + {{24{sram_act_rdata1[23]}}, sram_act_rdata1[23:16]} * {{24{sram_weight_rdata1[23]}}, sram_weight_rdata1[23:16]} 
                    + {{24{sram_act_rdata1[31]}}, sram_act_rdata1[31:24]} * {{24{sram_weight_rdata1[31]}}, sram_weight_rdata1[31:24]};
        end
        fc2_sum = fc2_psum + fc2_mac;
    end
    else begin
        fc2_mac = 0;
        fc2_sum = 0;
    end
end
//------------------------------------------------------------//

always @(posedge clk or negedge rst_n ) begin
    if (~rst_n)begin
        fc2_out_index <=  0;
    end
    else if (state == fc2_write)begin
        fc2_out_index <= fc2_out_index_tmp;
    end
    else begin
        fc2_out_index <=  0;
    end
end
always @* begin
    if (state == fc2_write)begin
        fc2_out_index_tmp = fc2_out_index + 2;
    end
    else begin
        fc2_out_index_tmp = 0;
    end
end

endmodule