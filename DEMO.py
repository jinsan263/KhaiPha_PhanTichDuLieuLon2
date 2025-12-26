# =====================================================
# SALES PROFIT PREDICTION APP – FINAL (SUBMISSION VERSION)
# Model: Random Forest Regressor
# =====================================================

from sklearn import metrics
import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.stats import gaussian_kde


# =====================================================
# APP CLASS
# =====================================================
class SalesProfitApp(ctk.CTk):
    def show_output(self):
        self.output.pack(fill="x", padx=10, pady=(10, 5))

    def hide_output(self):
        self.output.pack_forget()

    def __init__(self):
        super().__init__()

        self.title("PHÂN TÍCH VÀ DỰ ĐOÁN PROFIT BẰNG MÔ HÌNH RANDOM FOREST")
        self.geometry("1250x700")

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.df = None
        self.selected_features = []
        self.target_col = None
        self.model = None

        # ================= PANELS =================
        self.left_panel = ctk.CTkFrame(self, width=280, corner_radius=10)
        self.left_panel.pack(side="left", fill="y", padx=10, pady=10)

        self.mid_panel = ctk.CTkFrame(self, width=280, corner_radius=10)
        self.mid_panel.pack(side="left", fill="y", padx=10, pady=10)

        self.right_panel = ctk.CTkFrame(self, corner_radius=10)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.build_left_panel()
        self.build_mid_panel()
        self.build_right_panel()

        self.lock_analysis_buttons()

    # =====================================================
    # LOCK / UNLOCK
    # =====================================================
    def lock_analysis_buttons(self):
        self.btn_describe.configure(state="disabled")
        self.btn_train.configure(state="disabled")
        self.btn_importance.configure(state="disabled")
        self.btn_predict.configure(state="disabled")

    def unlock_after_train(self):
        self.btn_importance.configure(state="normal")
        self.btn_predict.configure(state="normal")
        self.btn_actual_pred.configure(state="normal")

    # =====================================================
    # RESET RIGHT PANEL (CHART + BUTTON)
    # =====================================================
    def reset_chart_area(self, show_save=False):
        # Xóa toàn bộ biểu đồ / card cũ
        for w in self.chart_frame.winfo_children():
            w.destroy()

        # Xóa nút cũ
        for w in self.chart_button_frame.winfo_children():
            w.destroy()

        # Ẩn / hiện frame nút lưu
        if show_save:
            self.chart_button_frame.pack(fill="x", pady=(0, 10))
        else:
            self.chart_button_frame.pack_forget()

    # =====================================================
    # LEFT PANEL – LOAD DATA
    # =====================================================
    def build_left_panel(self):
        ctk.CTkLabel(self.left_panel, text="DỮ LIỆU", font=("Arial", 20, "bold")).pack(pady=10)

        ctk.CTkButton(
            self.left_panel,
            text="📂 Tải file CSV",
            command=self.load_csv
        ).pack(pady=10)

        self.features_frame = ctk.CTkScrollableFrame(self.left_panel, height=520)
        self.features_frame.pack(fill="both", expand=True, pady=10)

        self.feature_checks = {}

    # =====================================================
    # MIDDLE PANEL – ACTIONS
    # =====================================================
    def build_mid_panel(self):
        ctk.CTkLabel(self.mid_panel, text="PHÂN TÍCH", font=("Arial", 20, "bold")).pack(pady=10)

        # self.btn_describe = ctk.CTkButton(
        #     self.mid_panel, text="📊 Phân tích mô tả ", command=self.describe_data
        # )
        self.btn_describe = ctk.CTkButton(
            self.mid_panel,
            text="📊 Phân tích mô tả ",
            command=self.describe_profit
        )

        # self.btn_describe.pack(pady=8)
        self.btn_describe.pack(pady=8)

        self.btn_train = ctk.CTkButton(
            self.mid_panel, text="🤖 Huấn luyện mô hình Random Forest", command=self.train_model
        )
        self.btn_train.pack(pady=8)

        self.btn_importance = ctk.CTkButton(
            self.mid_panel, text="⭐ Tầm quan trọng đặc trưng", command=self.show_importance
        )
        self.btn_importance.pack(pady=8)

        self.btn_actual_pred = ctk.CTkButton(
            self.mid_panel,
            text="📈 Thực tế với Dự đoán",
            command=self.show_actual_vs_pred
        )
        self.btn_actual_pred.pack(pady=8)
        self.btn_actual_pred.configure(state="disabled")

        self.btn_predict = ctk.CTkButton(
            self.mid_panel, text="💰 Dự đoán Profit", command=self.show_predict_card
        )
        self.btn_predict.pack(pady=8)

    # =====================================================
    # RIGHT PANEL – OUTPUT
    # =====================================================
    def build_right_panel(self):
        # ===== Log output =====
        self.output = ctk.CTkTextbox(self.right_panel, font=("Arial", 14), height=180)


        # ===== Frame chính =====
        self.chart_frame = ctk.CTkFrame(self.right_panel, corner_radius=10)
        self.chart_frame.pack(fill="both", expand=True, padx=10, pady=(5, 5))

        # ===== Frame riêng cho nút =====
        self.chart_button_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.chart_button_frame.pack(fill="x", pady=(0, 10))

    # =====================================================
    # LOAD CSV
    # =====================================================
    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return

        self.df = pd.read_csv(path)
        self.output.delete("1.0", "end")
        self.output.insert("end", f"Đã tải dữ liệu: {path}\nSố dòng: {len(self.df)}\n\n")

        for w in self.features_frame.winfo_children():
            w.destroy()

        ctk.CTkLabel(self.features_frame, text="Chọn biến độc lập (X):").pack(anchor="w")
        self.feature_checks = {}

        for col in self.df.columns:
            var = ctk.BooleanVar()
            chk = ctk.CTkCheckBox(self.features_frame, text=col, variable=var)
            chk.pack(anchor="w", padx=5)
            self.feature_checks[col] = var

        ctk.CTkLabel(self.features_frame, text="\nChọn biến phụ thuộc (y):").pack(anchor="w")
        self.target_var = ctk.StringVar()

        for col in self.df.columns:
            rb = ctk.CTkRadioButton(self.features_frame, text=col, variable=self.target_var, value=col)
            rb.pack(anchor="w", padx=5)

        self.btn_describe.configure(state="normal")
        self.btn_train.configure(state="normal")



    # =====================================================
    # DESCRIPTIVE ANALYSIS – PROFIT (EMBEDDED CHART)
    # =====================================================

    def describe_profit(self):
        if self.df is None:
            messagebox.showwarning("Chưa có dữ liệu", "Vui lòng tải file CSV trước")
            return

        self.hide_output()

        self.reset_chart_area(show_save=True)
        self.output.delete("1.0", "end")
        if self.df is None:
            return

        self.target_col = self.target_var.get()
        if self.target_col == "":
            messagebox.showerror("Lỗi", "Chưa chọn biến Profit!")
            return

        self.reset_chart_area(show_save=True)

        fig, ax = plt.subplots(figsize=(8, 5))

        data = self.df[self.target_col].dropna()

        # Histogram
        ax.hist(
            data,
            bins=100,
            density=True,  # để khớp với KDE
            alpha=0.6,
            edgecolor="black"
        )

        # Đường KDE (cái dây)
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 500)
        ax.plot(x_vals, kde(x_vals), linewidth=2)

        ax.set_title(f"Phân bố {self.target_col}", fontsize=14)
        ax.set_xlabel(self.target_col)
        ax.set_ylabel("Mật độ")

        fig.tight_layout(pad=2)

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

        def save_chart():
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")]
            )
            if path:
                fig.savefig(path, dpi=300, bbox_inches="tight")
                messagebox.showinfo("OK", "Đã lưu biểu đồ")

        ctk.CTkButton(
            self.chart_button_frame,
            text="💾 Lưu biểu đồ",
            width=180,
            command=save_chart
        ).pack()

        plt.close(fig)

    # =====================================================
    # TRAIN RANDOM FOREST

    # =====================================================
    def train_model(self):

        self.selected_features = [c for c, v in self.feature_checks.items() if v.get()]
        self.target_col = self.target_var.get()

        if not self.selected_features or self.target_col == "":
            messagebox.showerror("Lỗi", "Hãy chọn X và y")
            return

        X = self.df[self.selected_features]
        y = self.df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        self.y_test = y_test
        self.y_pred = preds

        mse = metrics.mean_squared_error(y_test, preds)
        rmse = mse ** 0.5
        r2 = metrics.r2_score(y_test, preds)

        self.hide_output()
        self.reset_chart_area(show_save=False)

        # ===== CARD KẾT QUẢ =====
        result_card = ctk.CTkFrame(
            self.chart_frame,
            corner_radius=18,
            fg_color="#F8FAFF"
        )
        result_card.pack(expand=True, padx=60, pady=60)

        # ===== TIÊU ĐỀ =====
        ctk.CTkLabel(
            result_card,
            text="🌲 KẾT QUẢ HUẤN LUYỆN RANDOM FOREST",
            font=("Arial", 22, "bold"),
            text_color="#1F3A8A"
        ).pack(pady=(25, 20))

        # ===== KPI FRAME (CHỈ TẠO 1 LẦN) =====
        kpi_frame = ctk.CTkFrame(
            result_card,
            corner_radius=14,
            fg_color="white"
        )
        kpi_frame.pack(padx=30, pady=10)

        # ===== MSE =====
        ctk.CTkLabel(
            kpi_frame,
            text="MSE",
            font=("Arial", 16, "bold")
        ).pack(pady=(15, 5))

        ctk.CTkLabel(
            kpi_frame,
            text=f"{mse:.3f}",
            font=("Arial", 26, "bold"),
            text_color="#7C2D12"
        ).pack(pady=(0, 10))

        # ===== RMSE =====
        ctk.CTkLabel(
            kpi_frame,
            text="RMSE",
            font=("Arial", 16, "bold")
        ).pack(pady=(10, 5))

        ctk.CTkLabel(
            kpi_frame,
            text=f"{rmse:.3f}",
            font=("Arial", 26, "bold"),
            text_color="#DC2626"
        ).pack(pady=(0, 10))

        # ===== R2 =====
        ctk.CTkLabel(
            kpi_frame,
            text="R² Score",
            font=("Arial", 16, "bold")
        ).pack(pady=(10, 5))

        ctk.CTkLabel(
            kpi_frame,
            text=f"{r2:.3f}",
            font=("Arial", 26, "bold"),
            text_color="#16A34A"
        ).pack(pady=(0, 20))

        # ===== FOOTER =====
        ctk.CTkLabel(
            result_card,
            text="✅ Huấn luyện mô hình hoàn tất\n",
            font=("Arial", 15),
            text_color="#065F46",
            justify="center"
        ).pack(pady=(20, 30))

        self.unlock_after_train()

        # # ===== RESET UI SAU HUẤN LUYỆN =====
        # self.reset_chart_area(show_save=False)
        #
        # done_label = ctk.CTkLabel(
        #     self.chart_frame,
        #     text="✅ Huấn luyện mô hình hoàn tất!\n\n"
        #          "👉 Bạn có thể xem:\n"
        #          "• Tầm quan trọng đặc trưng\n"
        #          "• Dự đoán Profit",
        #     font=("Arial", 16),
        #     justify="center"
        # )
        # done_label.pack(expand=True)


    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================
    def show_importance(self):
        self.hide_output()
        if self.model is None:
            messagebox.showerror("Lỗi", "Mô hình chưa được huấn luyện!")
            return

        self.reset_chart_area(show_save=True)

        fig, ax = plt.subplots(figsize=(8, 5))

        importance = self.model.feature_importances_

        # ===== SẮP XẾP THEO ĐỘ QUAN TRỌNG =====
        idx = np.argsort(importance)
        sorted_features = np.array(self.selected_features)[idx]
        sorted_importance = importance[idx]

        # ===== VẼ BAR =====
        bars = ax.barh(
            sorted_features,
            sorted_importance,
            height=0.6
        )

        # ===== HIỂN THỊ GIÁ TRỊ =====
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.002,
                bar.get_y() + bar.get_height() / 2,
                f"{width:.3f}",
                va="center",
                fontsize=10
            )

        ax.set_title("Tầm quan trọng đặc trưng ", fontsize=16, fontweight="bold")
        ax.set_xlabel("Mức độ quan trọng", fontsize=12)
        ax.set_ylabel("Đặc trưng", fontsize=12)

        ax.grid(axis="x", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        fig.tight_layout(pad=2)

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

        # ====== NÚT LƯU BIỂU ĐỒ ======
        def save_chart():
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")]
            )
            if path:
                fig.savefig(path, dpi=300, bbox_inches="tight")
                messagebox.showinfo("OK", "Đã lưu biểu đồ tầm quan trọng!")

        ctk.CTkButton(
            self.chart_button_frame,
            text="💾 Lưu biểu đồ",
            width=180,
            command=save_chart
        ).pack()

        plt.close(fig)

    def show_actual_vs_pred(self):
        self.hide_output()

        if self.model is None:
            messagebox.showerror("Lỗi", "Mô hình chưa được huấn luyện!")
            return

        self.reset_chart_area(show_save=True)

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(
            self.y_test,
            self.y_pred,
            alpha=0.6
        )

        min_val = min(self.y_test.min(), self.y_pred.min())
        max_val = max(self.y_test.max(), self.y_pred.max())

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            linestyle="--",
            linewidth=2
        )

        ax.set_xlabel("Lợi nhuận thực tế", fontsize=12)
        ax.set_ylabel("Lợi nhuận dự báo", fontsize=12)
        ax.set_title(
            "Lợi nhuận thực tế so với lợi nhuận dự đoán",
            fontsize=15,
            fontweight="bold"
        )

        ax.grid(alpha=0.4)

        fig.tight_layout(pad=2)

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

        # ===== NÚT LƯU BIỂU ĐỒ =====
        def save_chart():
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG Image", "*.png")]
            )
            if path:
                fig.savefig(path, dpi=300, bbox_inches="tight")
                messagebox.showinfo("OK", "Đã lưu biểu đồ!")

        ctk.CTkButton(
            self.chart_button_frame,
            text="💾 Lưu biểu đồ",
            width=180,
            command=save_chart
        ).pack()

        plt.close(fig)

    # =====================================================
    # PREDICT PROFIT
    # =====================================================
    def show_predict_card(self):
        self.hide_output()
        self.reset_chart_area(show_save=False)

        # ===== CARD CHÍNH =====
        card = ctk.CTkFrame(
            self.chart_frame,
            corner_radius=18,
            fg_color="#F8FAFF"
        )
        card.pack(expand=True, padx=80, pady=60)

        # ===== TIÊU ĐỀ =====
        ctk.CTkLabel(
            card,
            text="💰 DỰ BÁO PROFIT",
            font=("Arial", 22, "bold"),
            text_color="#1F3A8A"
        ).pack(pady=(25, 10))

        ctk.CTkLabel(
            card,
            text="Nhập dữ liệu đầu vào và xem kết quả dự báo",
            font=("Arial", 14),
            text_color="#475569"
        ).pack(pady=(0, 20))

        # ===== BODY =====
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="x", padx=30)

        # ===== CỘT INPUT (TRÁI) =====
        input_col = ctk.CTkFrame(body, fg_color="white", corner_radius=14)
        input_col.pack(side="left", fill="both", expand=True, padx=(0, 15))

        ctk.CTkLabel(
            input_col,
            text="📥 DỮ LIỆU ĐẦU VÀO",
            font=("Arial", 16, "bold")
        ).pack(pady=15)

        self.predict_entries = {}
        for f in self.selected_features:
            ctk.CTkLabel(input_col, text=f).pack(anchor="w", padx=20)
            ent = ctk.CTkEntry(input_col, height=32)
            ent.pack(fill="x", padx=20, pady=6)
            self.predict_entries[f] = ent

        # ===== CỘT KẾT QUẢ (PHẢI) =====
        result_col = ctk.CTkFrame(body, fg_color="white", corner_radius=14)
        result_col.pack(side="right", fill="both", expand=True, padx=(15, 0))

        ctk.CTkLabel(
            result_col,
            text="📊 KẾT QUẢ DỰ BÁO",
            font=("Arial", 16, "bold")
        ).pack(pady=15)

        self.result_label = ctk.CTkLabel(
            result_col,
            text="—",
            font=("Arial", 30, "bold"),
            text_color="#16A34A"
        )
        self.result_label.pack(pady=40)

        # ===== NÚT DỰ ĐOÁN =====
        def predict():
            try:
                vals = {f: float(self.predict_entries[f].get()) for f in self.selected_features}
                X_new = pd.DataFrame([vals])
                result = self.model.predict(X_new)[0]

                self.result_label.configure(
                    text=f"${result:,.2f}",
                    text_color="#16A34A"
                )
            except:
                self.result_label.configure(
                    text="Lỗi dữ liệu",
                    text_color="#DC2626"
                )

        ctk.CTkButton(
            card,
            text="🔮 DỰ BÁO PROFIT",
            height=44,
            font=("Arial", 16, "bold"),
            command=predict,
        ).pack(pady=25)


# =====================================================
# RUN APP
# =====================================================
if __name__ == "__main__":
    app = SalesProfitApp()
    app.mainloop()
