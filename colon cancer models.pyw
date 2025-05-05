import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn import linear_model, ensemble
from sklearn.neural_network import MLPClassifier

class PredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Distant Metastasis Prediction (All rights reserved by Legendm1a2@163.com)")
        
        # 初始化模型
        self.initialize_models()
        # 加载训练数据
        self.load_training_data()
        # 创建界面
        self.create_widgets()
    
    def initialize_models(self):
        """初始化预测模型"""
        self.LR = linear_model.LogisticRegression(C=0.1, class_weight=None, penalty='l1', solver='saga')
        self.GBC = ensemble.GradientBoostingClassifier(
            learning_rate=0.1, max_depth=3, max_features='sqrt',
            min_samples_leaf=2, min_samples_split=2, n_estimators=50, subsample=0.9)
        self.MLP = MLPClassifier(
            activation='relu', alpha=0.001, hidden_layer_sizes=(100,),
            learning_rate_init=0.001, max_iter=400, solver='adam')
        
        self.candidates = {'LR': self.LR, 'GBC': self.GBC, 'MLP': self.MLP}
        self.Father_threshold_list = [0.12083902166690041, 0.12037463203913083, 0.13754279268409053]
    
    def load_training_data(self):
        """加载训练数据（这里需要替换为实际的数据加载代码）"""
        # 示例代码，实际使用时需要替换为真实的数据加载方式
        try:
            self.data_training = pd.read_excel(r'D:\大强强的世界\python代码\colon cancer\Data.xlsx', 'Training Set')
            self.train_ftr = self.data_training.iloc[:, :8]
            self.train_tgt = self.data_training.iloc[:, -1]
        except Exception as e:
            print(f"加载训练数据时出错: {e}")
            # 创建空DataFrame作为后备
            self.train_ftr = pd.DataFrame(columns=['Age', 'T Stage', 'N Stage', 'RLNE', 
                                                 'Tumor Deposit', 'CEA', 'Perineural Invasion', 'Primary Site'])
            self.train_tgt = pd.Series()
    
    def create_widgets(self):
        """创建界面组件"""
        # 参数选项定义及对应的分类标签映射
        self.parameters = {
            "Age": {
                "options": ["20-39", "40-59", "60-79", ">80"],
                "mapping": {"20-39": 0, "40-59": 1, "60-79": 2, ">80": 3}
            },
            
            "Primary Site": {
                "options": ["other sites", "cecum"],
                "mapping": {"other sites": 0, "cecum": 1}
            },

            "T Stage": {
                "options": ["T1", "T2", "T3", "T4"],
                "mapping": {"T1": 0, "T2": 1, "T3": 2, "T4": 3}
            },
            "N Stage": {
                "options": ["N0", "N1", "N2"],
                "mapping": {"N0": 0, "N1": 1, "N2": 2}
            },
            "RLNE": {
                "options": ["≤12", ">12"],
                "mapping": {"≤12": 0, ">12": 1}
            },
            "Tumor Deposit": {
                "options": ["no tumor deposit found", "≥1 tumor deposits found"],
                "mapping": {"no tumor deposit found": 0, "≥1 tumor deposits found": 1}
            },
            "CEA": {
                "options": ["normal", "positive"],
                "mapping": {"normal": 0, "positive": 1}
            },
            "Perineural Invasion": {
                "options": ["not identified", "identified"],
                "mapping": {"not identified": 0, "identified": 1}
            }
        }
        
        # 存储所有下拉框的变量
        self.dropdown_vars = {}
        
        # 创建输入参数区域
        input_frame = ttk.LabelFrame(self.root, text="输入参数", padding=10)
        input_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        
        for i, (param_name, param_data) in enumerate(self.parameters.items()):
            # 创建标签
            label = ttk.Label(input_frame, text=f"{param_name}:")
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            
            # 创建下拉框
            var = tk.StringVar()
            var.set(param_data["options"][0])  # 设置默认值
            dropdown = ttk.Combobox(input_frame, textvariable=var, 
                                  values=param_data["options"], state="readonly")
            dropdown.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            
            # 存储变量
            self.dropdown_vars[param_name] = var
        
        # 创建按钮区域
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")
        
        submit_btn = ttk.Button(button_frame, text="运行预测", command=self.run_prediction)
        submit_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(button_frame, text="清空", command=self.clear_inputs)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 创建结果显示区域
        result_frame = ttk.LabelFrame(self.root, text="预测结果", padding=10)
        result_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        
        self.result_text = tk.Text(result_frame, height=10, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        # 配置网格布局权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)
        
        # 设置窗口最小大小
        self.root.minsize(400, 700)  # 增加了高度以容纳更多参数
    
    def clear_inputs(self):
        for param_name, var in self.dropdown_vars.items():
            default_option = self.parameters[param_name]["options"][0]
            var.set(default_option)
        self.result_text.delete(1.0, tk.END)
    
    def probs_to_prediction(self, probs, threshold):
        """将概率转换为预测结果"""
        return [1 if x > threshold else 0 for x in probs]
    
    def run_prediction(self):
        """运行预测"""
        try:
            # 收集用户输入并转换为编码
            encoded_values = {}
            for param_name, var in self.dropdown_vars.items():
                selected_text = var.get()
                encoded_values[param_name] = self.parameters[param_name]["mapping"][selected_text]
            
            # 创建测试数据DataFrame
            test_ftr = pd.DataFrame([encoded_values])
            
            # 清空结果区域
            self.result_text.delete(1.0, tk.END)
            
            # 运行各个模型的预测
            results = []
            for (model_name, model), set_thresh in zip(self.candidates.items(), self.Father_threshold_list):
                # 训练模型（在实际应用中，可能已经预训练好）
                model.fit(self.train_ftr, self.train_tgt)
                
                # 进行预测
                prob_true = model.predict_proba(test_ftr)[:, 1]
                preds = self.probs_to_prediction(prob_true, set_thresh)
                
                results.append(f"{model_name} 预测结果: {preds[0]} (概率: {prob_true[0]:.4f}, 阈值: {set_thresh:.4f})")
            
            # 显示结果
            self.result_text.insert(tk.END, "=== 预测结果 ===\n\n")
            self.result_text.insert(tk.END, "\n".join(results))
            self.result_text.insert(tk.END, "\n\n=== 输入参数 ===\n\n")
            
            # 显示输入参数
            input_params = []
            for param_name, var in self.dropdown_vars.items():
                input_params.append(f"{param_name}: {var.get()} (编码: {encoded_values[param_name]})")
            
            self.result_text.insert(tk.END, "\n".join(input_params))
            
        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"预测过程中出错:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()