import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import linregress

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置五号字体大小（10.5pt）
FONT_SIZE = 10.5
TITLE_SIZE = 12
LABEL_SIZE = 11
TICK_SIZE = 10


class ExtendedKLExperiment:
    def __init__(self):
        self.results = {}
        self.many_q_results = []  # 存储多个Q分布的结果

    def calculate_entropy(self, distribution):
        """计算分布的熵"""
        entropy = 0
        for p in distribution:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def calculate_kl_divergence(self, p_dist, q_dist):
        """计算KL散度D(P∥Q)"""
        kl = 0
        for i in range(len(p_dist)):
            if p_dist[i] > 0 and q_dist[i] > 0:
                kl += p_dist[i] * math.log2(p_dist[i] / q_dist[i])
        return kl

    def ideal_code_length(self, distribution):
        """理想码长（非整数）"""
        code_lengths = []
        for p in distribution:
            if p > 0:
                code_lengths.append(-math.log2(p))
            else:
                code_lengths.append(float('inf'))
        return code_lengths

    def shannon_code_length(self, distribution):
        """香农码码长（向上取整）"""
        code_lengths = []
        for p in distribution:
            if p > 0:
                code_lengths.append(math.ceil(-math.log2(p)))
            else:
                code_lengths.append(0)
        return code_lengths

    def generate_sequence(self, distribution, length):
        """根据分布生成符号序列"""
        symbols = list(range(len(distribution)))
        np.random.seed(42)
        sequence = np.random.choice(symbols, size=length, p=distribution)
        return sequence

    def calculate_average_length(self, sequence, code_lengths):
        """计算序列的平均码长"""
        if len(sequence) == 0:
            return 0

        # 统计符号出现频率
        symbol_counts = np.bincount(sequence)
        total_symbols = len(sequence)

        # 计算加权平均
        avg_length = 0
        for symbol, count in enumerate(symbol_counts):
            if count > 0:
                avg_length += (count / total_symbols) * code_lengths[symbol]

        return avg_length

    def run_extended_experiment(self, p_dist, q_dists_list, sequence_length=100000):
        """运行扩展实验：多个Q分布"""
        # 生成序列
        sequence = self.generate_sequence(p_dist, sequence_length)

        # 理论值计算
        H_P = self.calculate_entropy(p_dist)

        # 清空之前的结果
        self.many_q_results = []

        # 对每个错误分布Q进行实验
        for i, q_dist in enumerate(q_dists_list):
            # 使用错误分布Q的理想码长
            q_ideal_lengths = self.ideal_code_length(q_dist)

            # 计算平均码长
            L_Q_ideal = self.calculate_average_length(sequence, q_ideal_lengths)

            # 计算KL散度
            D_PQ = self.calculate_kl_divergence(p_dist, q_dist)

            # 存储结果
            self.many_q_results.append({
                'q_dist': q_dist,
                'D_PQ': D_PQ,
                'L_Q_ideal': L_Q_ideal,
                'redundancy': L_Q_ideal - H_P,
                'q1_value': q_dist[0]  # 记录Q的第一个分量，用于分类
            })

        return self.many_q_results

    def run_main_experiment(self, p_dist, main_q_dists, sequence_length=100000):
        """运行主实验：三个主要的Q分布"""
        # 生成序列
        sequence = self.generate_sequence(p_dist, sequence_length)

        # 理论值计算
        H_P = self.calculate_entropy(p_dist)

        # 使用真实分布P的理想码长和香农码码长
        p_ideal_lengths = self.ideal_code_length(p_dist)
        p_shannon_lengths = self.shannon_code_length(p_dist)

        L_P_ideal = self.calculate_average_length(sequence, p_ideal_lengths)
        L_P_shannon = self.calculate_average_length(sequence, p_shannon_lengths)

        # 存储结果
        self.results = {
            'p_dist': p_dist,
            'H_P': H_P,
            'L_P_ideal': L_P_ideal,
            'L_P_shannon': L_P_shannon,
            'q_results': []
        }

        # 对每个错误分布Q进行实验
        for i, q_dist in enumerate(main_q_dists):
            # 使用错误分布Q的理想码长和香农码码长
            q_ideal_lengths = self.ideal_code_length(q_dist)
            q_shannon_lengths = self.shannon_code_length(q_dist)

            # 计算平均码长
            L_Q_ideal = self.calculate_average_length(sequence, q_ideal_lengths)
            L_Q_shannon = self.calculate_average_length(sequence, q_shannon_lengths)

            # 计算KL散度
            D_PQ = self.calculate_kl_divergence(p_dist, q_dist)

            # 理论预测的平均码长（理想情况）
            L_Q_theoretical = H_P + D_PQ

            # 存储结果
            self.results['q_results'].append({
                'q_dist': q_dist,
                'D_PQ': D_PQ,
                'L_Q_ideal': L_Q_ideal,
                'L_Q_shannon': L_Q_shannon,
                'L_Q_theoretical': L_Q_theoretical,
                'rounding_error': L_Q_shannon - L_Q_ideal,
                'redundancy_ideal': L_Q_ideal - H_P,
                'redundancy_shannon': L_Q_shannon - H_P
            })

        return self.results

    def generate_many_q_distributions(self, p_dist, num_points=20):
        """生成多个错误分布Q"""
        q_dists = []

        # 第一种方法：在0.1到0.9之间均匀采样
        for i in range(num_points):
            # 确保Q不是P，并且Q的每个分量都不为0
            q1 = 0.1 + 0.8 * i / (num_points - 1)
            q1 = max(0.01, min(0.99, q1))  # 确保在(0,1)范围内
            q_dists.append([q1, 1 - q1])

        return q_dists

    def visualize_extended_results(self):
        """可视化扩展实验结果，调整字号避免重叠"""
        # 创建更大的图形以避免重叠
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 主实验：三个主要分布的结果
        ax1 = axes[0, 0]
        categories1 = []
        values1 = []

        categories1.extend(['H(P)', 'L_P(理想)'])
        values1.extend([self.results['H_P'], self.results['L_P_ideal']])

        for i, q_result in enumerate(self.results['q_results']):
            categories1.extend([
                f'H(P)+D(P∥Q{i + 1})',
                f'L_Q{i + 1}(理想)'
            ])
            values1.extend([
                q_result['L_Q_theoretical'],
                q_result['L_Q_ideal']
            ])

        # 调整条形图布局
        x_pos1 = np.arange(len(categories1))
        bars1 = ax1.bar(x_pos1, values1, width=0.5,
                        color=['blue', 'lightblue'] * (len(categories1) // 2))

        # 调整x轴标签位置和旋转角度
        ax1.set_xticks(x_pos1)
        ax1.set_xticklabels(categories1, rotation=25, ha='right', fontsize=FONT_SIZE)

        # 设置y轴标签和标题
        ax1.set_ylabel('比特数', fontsize=LABEL_SIZE)
        ax1.set_title('主实验：理想码长下理论值与实验值比较', fontsize=TITLE_SIZE, pad=15)

        # 设置网格和参考线
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=self.results['H_P'], color='red', linestyle='--', alpha=0.5, label='H(P)', linewidth=1.5)

        # 调整数值标签位置和大小
        for bar, value in zip(bars1, values1):
            height = bar.get_height()
            # 根据高度调整标签位置
            if height < max(values1) * 0.4:
                va_pos = 'bottom'
                y_offset = 0.02
            else:
                va_pos = 'top'
                y_offset = -0.02

            ax1.text(bar.get_x() + bar.get_width() / 2., height + y_offset,
                     f'{value:.3f}', ha='center', va=va_pos, fontsize=FONT_SIZE - 1,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                               edgecolor='none', alpha=0.8))

        ax1.legend(fontsize=FONT_SIZE, loc='upper left')

        # 2. 扩展实验：冗余与KL散度的关系（大量数据点）
        ax2 = axes[0, 1]

        # 从扩展实验中提取数据
        kl_values = [result['D_PQ'] for result in self.many_q_results]
        redundancy_values = [result['redundancy'] for result in self.many_q_results]
        q1_values = [result['q1_value'] for result in self.many_q_results]

        # 使用颜色映射表示Q的第一个分量值
        scatter = ax2.scatter(kl_values, redundancy_values,
                              c=q1_values, cmap='viridis', s=60, alpha=0.8,
                              edgecolors='black', linewidth=0.5)

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax2, pad=0.02)
        cbar.set_label('Q的第一个分量值', fontsize=LABEL_SIZE)
        cbar.ax.tick_params(labelsize=TICK_SIZE)

        # 绘制理想情况下的对角线
        min_val = min(min(kl_values), min(redundancy_values))
        max_val = max(max(kl_values), max(redundancy_values))
        padding = (max_val - min_val) * 0.05
        ax2.plot([min_val - padding, max_val + padding],
                 [min_val - padding, max_val + padding],
                 'r--', alpha=0.8, linewidth=2, label='理想线: 冗余 = KL散度')

        # 计算并绘制线性回归线
        if len(kl_values) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(kl_values, redundancy_values)
            regression_line = [slope * x + intercept for x in kl_values]
            ax2.plot(kl_values, regression_line, 'g-', alpha=0.7,
                     linewidth=2, label=f'回归线: y={slope:.4f}x+{intercept:.4f}')

            # 将R²值移到右上角，避免与Q1、Q2、Q3标签重叠
            r_squared = r_value ** 2

            # 修改这里：去掉方框，直接显示文本
            ax2.text(0.98, 0.98, f'$R^2$ = {r_squared:.6f}', transform=ax2.transAxes,
                     fontsize=FONT_SIZE, verticalalignment='top', horizontalalignment='right')

        # 标记三个主要实验点，减小点的大小，避免遮挡
        label_positions = {
            'Q1': {'offset_x': 0, 'offset_y': 0.015, 'ha': 'left', 'va': 'bottom'},
            'Q2': {'offset_x': 0, 'offset_y': -0.015, 'ha': 'right', 'va': 'top'},
            'Q3': {'offset_x': 0, 'offset_y': 0.015, 'ha': 'right', 'va': 'bottom'}
        }

        for i, q_result in enumerate(self.results['q_results']):
            label_key = f'Q{i + 1}'
            pos = label_positions[label_key]

            # 减小主实验点的大小，避免遮挡（从150减小到80）
            ax2.scatter(q_result['D_PQ'], q_result['redundancy_ideal'],
                        color='red', s=80, marker='s', edgecolors='black',
                        linewidth=1.5, zorder=5, label=f'主实验点{label_key}' if i == 0 else None)

            # 添加标签，使用不同偏移量避免重叠
            ax2.text(q_result['D_PQ'] + pos['offset_x'],
                     q_result['redundancy_ideal'] + pos['offset_y'],
                     label_key, fontsize=FONT_SIZE, ha=pos['ha'], va=pos['va'],
                     fontweight='bold', bbox=dict(boxstyle='round,pad=0.2',
                                                  facecolor='white', edgecolor='none', alpha=0.9))

        ax2.set_xlabel('KL散度 D(P∥Q)', fontsize=LABEL_SIZE)
        ax2.set_ylabel('理想冗余 (L_Q_ideal - H(P))', fontsize=LABEL_SIZE)
        ax2.set_title('扩展实验：冗余与KL散度的关系（20个数据点）', fontsize=TITLE_SIZE, pad=15)
        ax2.legend(fontsize=FONT_SIZE, loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 3. 误差分布图：展示冗余与KL散度的差异
        ax3 = axes[1, 0]
        differences = [abs(result['redundancy'] - result['D_PQ']) for result in self.many_q_results]
        kl_bins = np.linspace(min(kl_values), max(kl_values), 8)  # 减少分箱数量

        # 按KL散度大小分组计算平均差异
        bin_indices = np.digitize(kl_values, kl_bins)
        bin_avg_differences = []
        bin_centers = []
        for i in range(1, len(kl_bins)):
            mask = (bin_indices == i)
            if np.any(mask):
                bin_avg_differences.append(np.mean(np.array(differences)[mask]))
                bin_centers.append((kl_bins[i - 1] + kl_bins[i]) / 2)

        # 创建条形图
        bars3 = ax3.bar(bin_centers, bin_avg_differences,
                        width=(kl_bins[1] - kl_bins[0]) * 0.6,  # 进一步减小宽度
                        color='skyblue', edgecolor='black', alpha=0.7)

        # 将数据标签水平显示在柱状图上（旋转角度改为0）
        for bar, value in zip(bars3, bin_avg_differences):
            height = bar.get_height()
            if height > max(bin_avg_differences) * 0.1:  # 只显示较高的柱状图标签
                ax3.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{value:.4f}', ha='center', va='bottom',
                         fontsize=FONT_SIZE - 1, rotation=0)  # 旋转角度改为0（水平显示）

        ax3.set_xlabel('KL散度 D(P∥Q)', fontsize=LABEL_SIZE)
        ax3.set_ylabel('平均绝对差异', fontsize=LABEL_SIZE)
        ax3.set_title('冗余与KL散度差异的分布', fontsize=TITLE_SIZE, pad=15)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        # 4. Q分布参数与KL散度的关系
        ax4 = axes[1, 1]

        # 按Q的第一个分量排序
        sorted_indices = np.argsort(q1_values)
        sorted_q1 = np.array(q1_values)[sorted_indices]
        sorted_kl = np.array(kl_values)[sorted_indices]

        # 绘制曲线
        ax4.plot(sorted_q1, sorted_kl, 'b-', linewidth=2, alpha=0.7)

        # 标记关键点（两端和中间各一个），避免过多标签
        key_indices = [0, len(sorted_q1) // 2, len(sorted_q1) - 1]
        for idx in key_indices:
            x, y = sorted_q1[idx], sorted_kl[idx]
            # 根据点的位置调整标签位置
            if idx == 0:
                ha_pos, va_pos = 'left', 'bottom'
            elif idx == len(sorted_q1) // 2:
                ha_pos, va_pos = 'center', 'top'
            else:
                ha_pos, va_pos = 'right', 'bottom'

            ax4.text(x, y, f'({x:.2f},{y:.3f})', fontsize=FONT_SIZE - 2,
                     ha=ha_pos, va=va_pos,
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

        ax4.set_xlabel('Q的第一个分量值', fontsize=LABEL_SIZE)
        ax4.set_ylabel('KL散度 D(P∥Q)', fontsize=LABEL_SIZE)
        ax4.set_title('Q分布参数与KL散度的关系', fontsize=TITLE_SIZE, pad=15)
        ax4.grid(True, alpha=0.3)

        # 标记P的第一个分量值，将图例移到右上角避免遮挡
        p1 = self.results['p_dist'][0]
        ax4.axvline(x=p1, color='red', linestyle='--', alpha=0.7,
                    linewidth=1.5, label=f'P的第一个分量={p1}')
        ax4.legend(fontsize=FONT_SIZE, loc='upper right')  # 改为右上角

        # 设置所有坐标轴的刻度字体大小
        for ax in [ax1, ax2, ax3, ax4]:
            ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE)

        # 调整整体布局，避免重叠
        plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)

        return fig

    def print_statistical_analysis(self):
        """打印统计分析结果"""
        print("=" * 50)
        print("扩展实验：统计分析结果")

        # 从扩展实验中提取数据
        kl_values = [result['D_PQ'] for result in self.many_q_results]
        redundancy_values = [result['redundancy'] for result in self.many_q_results]

        # 计算基本统计量
        mean_kl = np.mean(kl_values)
        std_kl = np.std(kl_values)
        mean_red = np.mean(redundancy_values)
        std_red = np.std(redundancy_values)

        print(f"KL散度统计: 均值={mean_kl:.6f}, 标准差={std_kl:.6f}")
        print(f"冗余统计: 均值={mean_red:.6f}, 标准差={std_red:.6f}")

        # 计算差异统计
        differences = [abs(r - k) for r, k in zip(redundancy_values, kl_values)]
        mean_diff = np.mean(differences)
        max_diff = np.max(differences)

        print(f"冗余与KL散度差异: 平均绝对差异={mean_diff:.6f}, 最大差异={max_diff:.6f}")

        # 线性回归分析
        if len(kl_values) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(kl_values, redundancy_values)
            r_squared = r_value ** 2

            print(f"\n线性回归分析:")
            print(f"  斜率: {slope:.6f} (理论应为1.0)")
            print(f"  截距: {intercept:.6f} (理论应为0.0)")
            print(f"  相关系数R: {r_value:.6f}")
            print(f"  决定系数R²: {r_squared:.6f}")
            print(f"  P值: {p_value:.6e}")
            print(f"  标准误差: {std_err:.6f}")

            # 假设检验
            alpha = 0.05
            if p_value < alpha:
                print(f"  结果: 拒绝原假设，冗余与KL散度存在显著线性关系 (p < {alpha})")
            else:
                print(f"  结果: 无法拒绝原假设，冗余与KL散度无显著线性关系 (p ≥ {alpha})")

            # 置信区间
            from scipy import stats
            t_critical = stats.t.ppf(1 - alpha / 2, len(kl_values) - 2)
            ci_slope = (slope - t_critical * std_err, slope + t_critical * std_err)
            print(f"  斜率95%置信区间: ({ci_slope[0]:.6f}, {ci_slope[1]:.6f})")

            # 检查斜率是否包含1
            if ci_slope[0] <= 1 <= ci_slope[1]:
                print(f"  结论: 置信区间包含1，支持冗余=KL散度的理论")
            else:
                print(f"  结论: 置信区间不包含1，与理论有偏差")


# 运行扩展实验
def main():
    print("实验四：冗余 = KL散度")
    print("=" * 50)

    # 设置分布参数
    P = [0.7, 0.3]  # 真实分布

    # 三个主要的Q分布（用于对比）
    main_q_dists = [
        [0.5, 0.5],  # 均匀分布
        [0.9, 0.1],  # 极端分布
        [0.8, 0.2]  # 中度偏离分布
    ]

    # 创建实验实例
    experiment = ExtendedKLExperiment()

    # 1. 运行主实验（三个主要分布）
    print("运行主实验（三个主要分布）...")
    results = experiment.run_main_experiment(P, main_q_dists, sequence_length=100000)

    # 2. 生成多个Q分布
    print("生成多个Q分布...")
    many_q_dists = experiment.generate_many_q_distributions(P, num_points=20)

    # 3. 运行扩展实验
    print("运行扩展实验（20个分布）...")
    extended_results = experiment.run_extended_experiment(P, many_q_dists, sequence_length=100000)

    # 4. 打印统计分析
    experiment.print_statistical_analysis()

    # 5. 可视化结果
    print("\n生成可视化图表...")
    fig = experiment.visualize_extended_results()

    # 保存图表
    plt.savefig('extended_kl_experiment_final.png', dpi=300, bbox_inches='tight')
    print("图表已保存为 'extended_kl_experiment_final.png'")

    plt.show()


if __name__ == "__main__":
    main()