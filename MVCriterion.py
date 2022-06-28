import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

class PORT:
    def __init__(self, stocks, port_size, dates):
        self.stocks = stocks
        self.port_size = port_size
        
        self.w = np.repeat(1/self.port_size, self.port_size)
        
        self.dates = dates
        start_date, end_date = dates.split(":")
        start_date = pd.Timestamp(start_date).date()
        end_date = pd.Timestamp(end_date).date()
        from itertools import combinations
        self.x = list(combinations(stocks, self.port_size))
        self.y = [' '.join(i) for i in self.x]
        
        print(start_date, end_date)

        print("Loading data...")
        self.df = pdr.get_data_yahoo(stocks, start_date, end_date)
        print("Done!")
        
        self.df = self.df['Adj Close']
        self.df = self.df.dropna(inplace=False)
        self.returns = self.df.pct_change()
        self.returns = self.returns.dropna(inplace=False)
        self.mean_return = self.returns.mean()
        self.std = self.returns.std()
        
        self.correlations = self.returns.corr()
        self.p_return = np.zeros(len(self.y))
        i=0
        while i <= len(self.y)-1:
            self.q = self.y[i].split(" ")
            self.pi_return_vec = np.zeros(self.port_size)
            j=0
            while j <= self.port_size-1:
                self.pi_return_vec[j] = self.mean_return[self.q[j]]
                j = j+1
            self.pi_return = (1/self.port_size)*sum(self.pi_return_vec)
            self.p_return[i] = self.pi_return
            i = i+1
        
        
        self.cov_matrix = self.returns.cov()
        self.p_std = np.zeros(len(self.y))
        i=0
        while i <= len(self.y)-1:
            self.q = self.y[i].split(" ")
            self.cov_matrix_i = self.returns[self.q].cov()
            self.pi_std = np.sqrt(np.dot(self.w.T, np.dot(self.cov_matrix_i,self.w)))
            self.p_std[i] = self.pi_std
            i = i+1
            
        self.dreturns = dict()
        i=0
        while i <= len(self.y)-1:
            self.dreturns[self.p_return[i]] = self.y[i]
            i = i+1
        self.dstd = dict()
        i=0
        while i <= len(self.y)-1:
            self.dstd[self.p_std[i]] = self.y[i]
            i = i+1
        
    def price_over_time(self):
        self.df.plot(figsize=(16,9))
        plt.title("Price over time", fontdict={'family': 'serif', 'color' : 'black','weight': 'bold','size': 18})
        plt.xlabel("Date", fontsize=16, fontweight='bold')
        plt.ylabel("Price", fontsize=16, fontweight='bold')
        
    def return_over_time(self):
        self.returns.plot(figsize=(16,9))
        plt.title("Return over time", fontdict={'family': 'serif', 'color' : 'black','weight': 'bold','size': 18})
        plt.xlabel("Date", fontsize=16, fontweight='bold')
        plt.ylabel("Return", fontsize=16, fontweight='bold')
        
    def stocks_mv(self):
        if min(self.std)<0 and max(self.std)<0:
            plt.xlim(min(self.std)-0.0025,+0.0025)
        elif min(self.std)>0 and max(self.std)>0:
            plt.xlim(-0.0025,max(self.std)+0.0025)
        else: plt.xlim(min(self.std)-0.0025,max(self.std)+0.0025)
                
        if min(self.mean_return)<0 and max(self.mean_return)<0:
            plt.ylim(min(self.mean_return)-0.0004,+0.0004)
        elif min(self.mean_return)>0 and max(self.mean_return)>0:
            plt.ylim(-0.0004,max(self.mean_return)+0.0004)
        else: plt.ylim(min(self.mean_return)-0.0004,max(self.mean_return)+0.0004)
        i=0
        while i<=len(self.stocks)-1:
            plt.text(self.std[i], self.mean_return[i], self.stocks[i], fontsize=14, fontweight='bold', color="black",
                     ha='center', va='center')
            i=i+1
        plt.plot([-1,+1],[0,0], color="green")
        plt.plot([0,0],[-1,+1], color="green")
        plt.title("Mean-Variance Criterion", fontdict={'family': 'serif', 'color' : 'black','weight': 'bold','size': 20})
        plt.xlabel("Standard deviation", fontsize=18, fontweight='bold')
        plt.ylabel("Mean return", fontsize=18, fontweight='bold')
        plt.rcParams["figure.figsize"] = (10,10)
        plt.show()
        
    def stocks_correlation(self):
        sns.set(rc = {'figure.figsize':(20,10)})
        sns.heatmap(self.correlations, annot=True, cmap='RdYlGn', vmax=1.0, vmin=-1.0)
        plt.title("Correlation between assets", fontdict={'family': 'serif', 'color' : 'black','weight': 'bold','size': 20})
        
    def portfolio_statistics(self):
        i = 1
        while i <= len(self.y):
            print('\033[1m' + "Portfolio ", end="")
            print(i, end="")
            print(" statistics ", end="")
            print(self.y[i-1], end="")
            print(":" + '\033[0m')
            print(" -Mean return: ", end="")
            print(round(self.p_return[i-1], 4))
            print(" -Standard deviation: ", end="")
            print(round(self.p_std[i-1], 4))
            i = i+1
        
    def portfolio_mv(self):
        df = pd.DataFrame(list(zip(self.y, self.p_std, self.p_return)), columns =['   P', 'Standard deviation', 'Return'])
        fig = px.scatter(df,x="Standard deviation", y="Return", color="   P", title="<b>Mean-Variance Criterion</b>", 
                         labels={"Return": "<b>Return</b>", "Standard deviation": "<b>Standard deviation</b>", 
                                 "   P": "<b>   P</b>"})
        fig.update_layout(font_color="black",title_font_family="serif",title_font_color="black",title_font_size=20, title_x=0.3)
        fig.add_hline(y=0)
        fig.add_vline(x=0)
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
        fig.show()
        
    def conclusions(self):
        print("The portfolio with maximum expected return is ", end="")
        print(self.dreturns[max(self.p_return)], end="")
        print(" and it's value is ", end="")
        print(round(max(self.p_return),5))
        
        print("The portfolio with minimum standard deviation is ", end="")
        print(self.dstd[min(self.p_std)], end="")
        print(" and it's value is ", end="")
        print(round(min(self.p_std),5))
        
        print("The portfolio with minimum expected return is ", end="")
        print(self.dreturns[min(self.p_return)], end="")
        print(" and it's value is ", end="")
        print(round(min(self.p_return),5))
        
        print("The portfolio with maximum standard deviation is ", end="")
        print(self.dstd[max(self.p_std)], end="")
        print(" and it's value is ", end="")
        print(round(max(self.p_std),5))

        
