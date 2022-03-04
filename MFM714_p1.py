
from msilib.schema import InstallUISequence
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from math import exp
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import datetime
import math
### Part 1 code here


###
### Code to deal with years passed
# years = 100
# days_per_year = 365.24
# hundred_years_later = my_object.date + datetime.timedelta(days=(years*days_per_year))
### TODO Upload the data
issuers =  pd.read_excel("data/Issuers_for_project_2022.xls",index_col=0)
portfolio = pd.read_excel("data/Portfolio_for_project_2022.xls",index_col=0)
transition = pd.read_csv("data/transition TBD.csv")
yield_curve = pd.read_excel("data/Yield curve for project 2022.xlsx",index_col=0, skiprows=[0], header= [1])


Cor_Matrix =  np.diag([1 for i in range(portfolio.shape[0])])
Cor_Matrix= np.where(Cor_Matrix == 0,0.36,Cor_Matrix) 

Chol_L = cholesky(Cor_Matrix)


### Part 2 code here
def market_val_port(port):
    l = port.Price * port.Notional /100
    l.columns = ["Name","Portfolio Value"]
    return(l)

def routine_price(portfolio,yc):
    # Set index in years to help make it easier for interpolation
    yc = yc.set_index("In years")
    """
    Description:
    Routine_price simulates the values of each bond and CDS relative to change in credit rating returning
    the simulated prices based on a transition matrix, and yield associated with the rating.

    Purpose:
    Using this procedure will reflect the price of the financial instrument based on credit rather for each credit rating.
    after a 1 year time step
    Variates:
    Routine_price: 
    - objtype: pd.DataFrame 
    - dtype: pd.float64
    - Matrix containing prices for each instrument.
    - Columns: The credit rating
    - rows: The  instrument and its prices at every Rating after passing 1 year
    
    portfolio: 
    - objtype: pd.DataFrame
    - dtype: pd.float64
    - Matrix containing portfolio of financial instruments and their associated information.
    
    yc:
    - objtype: pd.DataFrame
    - dtype: pd.float64
    - Matrix containing yields of bonds with respect to Credit ratings.
    """
    ### Todays date
    now_w = datetime.datetime.now()
    ## TODO Create routine such that for defaults result in Recovery value for Bonds, and Notional * recovery %
    # def makehams_formula(coup, rate, time_remaining, freq, Notional):
    #     ye = math.floor(time_remaining)
    #     adj = time_remaining - ye
    #     ## TODO adjust adj  such that we only need to consider the period after coupon.
        
    #     if adj > 0.5:
    #         time_to_maturity = ye + 0.5 
    #         adj  = (adj- 0.5)/0.5
    #     else:
    #         time_to_maturity = ye 
    #         adj = adj / 0.5
    #     # D is the discounting factor
    #     D = (1/(1+ rate/freq)**(freq* time_to_maturity))
    #     # P is the Clean price of the bond at the specific 
    #     Dirty = coup/rate * Notional *(1 - D)  + Notional * D
    #     # Dirty_price represents the Dirty Price of the bond
    #     P = Dirty * (1 + rate/2)**adj  -   Notional*coup*adj
    #     return(P, Dirty) 




    def makehams_formula(coup, rate, time_remaining, freq, Notional):
        ye = np.floor(time_remaining)
        adj = time_remaining - ye
        t2 = ye.copy()
        print(adj)
        ## TODO adjust adj  such that we only need to consider the period after coupon.
        def check_adj(adj):
            if adj > 0.5:
                #time_to_maturity = ye + 0.5 
                adj  = (adj- 0.5)/0.5
            else:
                
                adj = adj / 0.5
            return(adj)
        def check_adj_2(adj):
            if adj > 0.5:
                #time_to_maturity = ye + 0.5 
                k =  0.5
            else:
                k = 0
            return(k)
        

        t1 = adj.copy()

        adjust = adj.apply(lambda x: check_adj(x)) 
        time_to_maturity = ye + t1.apply(lambda x: check_adj_2(x))
        

        # D is the discounting factor
        D = (1/(1+ rate/freq)**(freq*time_to_maturity))
        # P is the Clean price of the bond at the specific 

        
        Dirty = coup/rate * Notional *(1 - D)  + Notional * D
        # Dirty_price represents the Dirty Price of the bond
        P = Dirty * (1 + rate/2)**adjust  -   Notional*coup*adjust
        return(P, Dirty) 
    
    # time_1 = now_w + timedelta(365.25)
    def CDS_pricer(status,Instrument,recovery,Notional):
        if Instrument == "CDS":
            if  status == "Default":
                return(Notional * recovery)
            else:
                return(0)


            
    def run_pricer(port):
    # TODO create a condition for CDSs and one for Bonds
        pass 
            

    # def one_step(port, rating):
    col_names = yc.columns
    col_names.remove("government","In years")
    ret = portfolio.copy()
    ret["Years_Remaining"] =  ret["Maturity_Date"].apply(lambda x: (relativedelta(x,now_w).years +relativedelta(x,now_w).months/12)) - 1
    result = pd.DataFrame(columns=col_names)
    #ret["Instrument"] = ret["Instrument type (Bond or CDS)"].apply(lambda x: 1 if x  == "Bond")
    #TODO check if "In years" contains else interperolate
    for i in col_names: 
        r = yc.loc()
        result[i] = makehams_formula(result.Coupon, )
    pass
    





if __name__ == "__main__":
    print("")