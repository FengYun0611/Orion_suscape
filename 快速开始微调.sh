#!/bin/bash
# ========================================================================
# SUScape微调快速开始脚本
# ========================================================================
# 
# 用法:
#   chmod +x 快速开始微调.sh
#   ./快速开始微调.sh
#
# 或指定GPU数量:
#   ./快速开始微调.sh 8  # 使用8个GPU
# ========================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置
NUM_GPUS=${1:-4}  # 默认4个GPU，可通过第一个参数指定
CONFIG="adzoo/orion/configs/suscape_finetune.py"
WORK_DIR="work_dirs/orion_suscape_finetune"
PRETRAINED_CKPT="ckpts/orion_stage3.pth"
DATA_DIR="data/suscape_infos"

# 打印带颜色的消息
print_header() {
    echo -e "${BLUE}========================================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# 检查环境
check_environment() {
    print_header "步骤 0: 检查环境"
    
    # 检查Python
    if command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        print_success "Python版本: $PYTHON_VERSION"
    else
        print_error "未找到Python"
        exit 1
    fi
    
    # 检查PyTorch
    if python -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
        print_success "PyTorch版本: $TORCH_VERSION"
        
        # 检查CUDA
        if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
            CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
            GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
            print_success "CUDA版本: $CUDA_VERSION"
            print_success "可用GPU: $GPU_COUNT 个"
            
            if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
                print_warning "可用GPU ($GPU_COUNT) 少于请求的GPU数量 ($NUM_GPUS)"
                print_info "将使用 $GPU_COUNT 个GPU"
                NUM_GPUS=$GPU_COUNT
            fi
        else
            print_warning "CUDA不可用，将使用CPU训练（不推荐）"
        fi
    else
        print_error "未安装PyTorch"
        exit 1
    fi
    
    echo ""
}

# 检查数据文件
check_data() {
    print_header "步骤 1: 检查数据文件"
    
    # 检查数据集pkl文件
    if [ -f "$DATA_DIR/suscape_infos_test.pkl" ]; then
        print_success "找到数据集文件: $DATA_DIR/suscape_infos_test.pkl"
    else
        print_error "未找到数据集文件: $DATA_DIR/suscape_infos_test.pkl"
        print_info "请确保已运行数据准备脚本"
        exit 1
    fi
    
    # 检查预训练模型
    if [ -f "$PRETRAINED_CKPT" ]; then
        print_success "找到预训练模型: $PRETRAINED_CKPT"
    else
        print_error "未找到预训练模型: $PRETRAINED_CKPT"
        print_info "请下载ORION Stage 3预训练权重"
        exit 1
    fi
    
    # 检查配置文件
    if [ -f "$CONFIG" ]; then
        print_success "找到配置文件: $CONFIG"
    else
        print_error "未找到配置文件: $CONFIG"
        exit 1
    fi
    
    echo ""
}

# 验证数据加载
validate_data() {
    print_header "步骤 2: 验证数据加载"
    
    if [ -f "validate_collision_detection.py" ]; then
        print_info "运行数据验证..."
        if python validate_collision_detection.py 2>&1 | grep -q "ALL VALIDATIONS PASSED"; then
            print_success "数据验证通过"
        else
            print_warning "数据验证有警告，但可以继续"
            print_info "详细信息请查看 validate_collision_detection.py 输出"
        fi
    else
        print_warning "未找到验证脚本，跳过验证步骤"
    fi
    
    echo ""
}

# 准备训练数据
prepare_train_data() {
    print_header "步骤 3: 准备训练数据"
    
    if [ -f "$DATA_DIR/suscape_infos_train.pkl" ] && [ -f "$DATA_DIR/suscape_infos_val.pkl" ]; then
        print_success "训练/验证数据已存在，跳过分割"
        
        # 显示数据统计
        TRAIN_SAMPLES=$(python -c "import pickle; data=pickle.load(open('$DATA_DIR/suscape_infos_train.pkl','rb')); print(len(data['infos']))")
        VAL_SAMPLES=$(python -c "import pickle; data=pickle.load(open('$DATA_DIR/suscape_infos_val.pkl','rb')); print(len(data['infos']))")
        print_info "训练样本: $TRAIN_SAMPLES"
        print_info "验证样本: $VAL_SAMPLES"
    else
        print_info "分割数据集 (80% 训练, 20% 验证)..."
        python tools/split_suscape_data.py \
            --input "$DATA_DIR/suscape_infos_test.pkl" \
            --output-dir "$DATA_DIR" \
            --train-ratio 0.8 \
            --seed 42
        
        if [ $? -eq 0 ]; then
            print_success "数据分割完成"
        else
            print_error "数据分割失败"
            exit 1
        fi
    fi
    
    echo ""
}

# 开始训练
start_training() {
    print_header "步骤 4: 开始微调训练"
    
    print_info "配置:"
    print_info "  - GPU数量: $NUM_GPUS"
    print_info "  - 配置文件: $CONFIG"
    print_info "  - 工作目录: $WORK_DIR"
    print_info "  - 预训练模型: $PRETRAINED_CKPT"
    echo ""
    
    # 创建工作目录
    mkdir -p "$WORK_DIR/logs"
    
    print_info "启动分布式训练..."
    print_warning "训练将需要几个小时到几天时间，取决于GPU性能"
    print_info "可以在另一个终端使用以下命令监控训练:"
    print_info "  tail -f $WORK_DIR/logs/train.*.log"
    echo ""
    
    # 等待用户确认
    read -p "按Enter键开始训练，或按Ctrl+C取消..." 
    
    if [ $NUM_GPUS -gt 1 ]; then
        # 多GPU训练
        ./adzoo/orion/orion_dist_train.sh \
            "$CONFIG" \
            $NUM_GPUS \
            --work-dir "$WORK_DIR" \
            --load-from "$PRETRAINED_CKPT" \
            --seed 42
    else
        # 单GPU训练
        python adzoo/orion/train.py \
            "$CONFIG" \
            --work-dir "$WORK_DIR" \
            --load-from "$PRETRAINED_CKPT" \
            --seed 42
    fi
    
    TRAIN_EXIT_CODE=$?
    echo ""
    
    if [ $TRAIN_EXIT_CODE -eq 0 ]; then
        print_success "训练完成！"
    else
        print_error "训练过程出错（退出码: $TRAIN_EXIT_CODE）"
        print_info "请检查日志: $WORK_DIR/logs/train.*.log"
        exit 1
    fi
}

# 显示后续步骤
show_next_steps() {
    print_header "下一步操作"
    
    echo "训练已完成！以下是后续操作建议:"
    echo ""
    
    print_info "1. 评估微调后的模型:"
    echo "   python adzoo/orion/test.py \\"
    echo "       adzoo/orion/configs/suscape_eval.py \\"
    echo "       $WORK_DIR/latest.pth \\"
    echo "       --eval bbox"
    echo ""
    
    print_info "2. 对比预训练模型和微调模型:"
    echo "   # 评估预训练模型"
    echo "   python adzoo/orion/test.py \\"
    echo "       adzoo/orion/configs/suscape_eval.py \\"
    echo "       $PRETRAINED_CKPT \\"
    echo "       --eval bbox"
    echo ""
    echo "   # 评估微调模型"
    echo "   python adzoo/orion/test.py \\"
    echo "       adzoo/orion/configs/suscape_eval.py \\"
    echo "       $WORK_DIR/latest.pth \\"
    echo "       --eval bbox"
    echo ""
    
    print_info "3. 查看训练日志:"
    echo "   cat $WORK_DIR/logs/train.*.log"
    echo ""
    
    print_info "4. 启动TensorBoard查看训练曲线:"
    echo "   tensorboard --logdir $WORK_DIR"
    echo ""
    
    print_info "5. 如需继续训练:"
    echo "   ./adzoo/orion/orion_dist_train.sh \\"
    echo "       $CONFIG \\"
    echo "       $NUM_GPUS \\"
    echo "       --resume-from $WORK_DIR/latest.pth"
    echo ""
    
    print_success "恭喜！微调流程已完成 🎉"
}

# 主流程
main() {
    clear
    
    print_header "SUScape数据集 ORION模型微调 - 快速开始"
    echo ""
    echo "本脚本将自动完成以下步骤:"
    echo "  0. 检查环境"
    echo "  1. 检查数据文件"
    echo "  2. 验证数据加载"
    echo "  3. 准备训练数据（分割为train/val）"
    echo "  4. 启动微调训练"
    echo ""
    echo "GPU数量: $NUM_GPUS"
    echo ""
    
    read -p "按Enter键继续，或按Ctrl+C取消..." 
    echo ""
    
    check_environment
    check_data
    validate_data
    prepare_train_data
    start_training
    show_next_steps
}

# 运行主流程
main
