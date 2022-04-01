import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from math import exp
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import datetime
import os

#Step 1 part i)


def upload_data(data_url="./data"):
    """
    upload_data 
    Takes in a file path and returns the 
    The ratings, portfolio, yield curve, and transition matrix
    as a pandas.DataFrame
    upload_data: str -> pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame
    """
    
    rate = os.path.join(data_url,"Issuers_for_project_2022.xls")
    port = os.path.join(data_url,"Portfolio_for_project_2022.xls")
    yc = os.path.join(data_url,"Yield curve for project 2022.xlsx")
    trans = os.path.join(data_url, "Transition Matrix.xlsx")
    
    # Read in data
    ratings = pd.read_excel(rate)
    portfolio =   pd.read_excel(port, index_col=0)
    yield_curve = pd.read_excel(yc,index_col=1,skiprows=[0], header= [1])
    transition =  pd.read_excel(trans,index_col=0, skiprows=[0])

    return( ratings, portfolio, yield_curve, transition)

def Correlation_Matrix(port):
    """
    Correlation_Matrix consume  a pd.DataFrame of portfolio positions
    and returns the associated correaltion matrix for each instrument

    Remark: 

    At the  moment correaltion matrix is filled on the diagnonal with 1s and
    0.36 The code will be changed at a later date
    """

    Cor_Matrix = np.diag([1 for i in range(port.shape[0])])
    Cor_Matrix = np.where(Cor_Matrix == 0,0.36, Cor_Matrix)
    return(Cor_Matrix)

def mkt_value(port):
    """
    market_value is a function that takes a pd.DataFrame of portfolio and returns the market value
    market_value: pd.DataFrame ->  pd.DataFrame
    Remark: At the moment the theortical clean price is the price, we will make the assumption of the 
    the proper price later on
    """
    port["Market Value"] =  port["Notional"] * port["Theoretical Clean Price"] / 100
    Total = port["Market Value"].sum()
    print("Total portfolio Value is: {}".format(Total))
    return(port)

def one_year_forward(port,transition):
    Step3 = pd.DataFrame(
    np.column_stack([port["Theoretical Clean Price"].values] * len(range(7))),
    port.index, transition.index)
    Step3["D"] = port["Expected Recovery rate"] * port["Notional"] /100
    Step3["Instrument type (Bond or CDS)"] = port["Instrument type (Bond or CDS)"]

    Step3 = Step3.reset_index().set_index(["Name", "Instrument type (Bond or CDS)"])

    return(Step3)


def ratings_sim(port, ratings, N=2000):
    #portfolio_w_ratings=portfolio_w_ratings.reset_index()
    portfolio_w_ratings = pd.merge(port, ratings, left_on=["Name"], right_on=["Name"])
    Step4 = pd.DataFrame(
        np.column_stack([portfolio_w_ratings["Issuer Rating"].values] * len(range(N))),
        port.index)
    Step4["Instrument type (Bond or CDS)"] = port["Instrument type (Bond or CDS)"]   
    
    #Step4 = Step4.set_index(["Name","Instrument type (Bond or CDS)"])
    
    return(Step4)

def pricing_matrix(simulation,lookup_hash):

    """
    pricing_matrix consumes the simulation pd.DataFrame and a lookup_hash which 
    contains all the potential prices for each scenario for each instrument
    
    Key note is that hte hash table is find via using pd.DataFrame.to_dict("records")
    The choice of hash table is to ensure that we can loop through each instrument. 
    """
    N = simulation.shape[0]
    M = simulation.shape[1]
    
    results = np.zeros(shape = simulation.shape)

    #print(results)
    i = 0 
    j = 0

    for labels,content in simulation.items():
        # grabs all the simulated Ratings
        for  s,r  in content.items():
            # loops through all the ratings and companys
            # sets hash such that the specific instrument with specific rating
            
            results[i,j] = lookup_hash[i][r]
            i += 1 
        i=0 
        j+=1
    
    return results   

def Mc_price(price,notional):
    """
    
    pricing_matrix takes in the simulated prices pd.DataFrame named "price" of the instruments
    and multiplies the the notional vector then the market value of each scenario as a pd.DataFrame
    
    In general:
     - Improvements must be made to find the market value and include cases for default
    """
    results = np.multiply(np.multiply(notional,price.T).T,(1/100))
    results = pd.DataFrame(results)
    return(results)

def Calculate_Statistics(numpy_results, intial_value):
    """
    Calculate_Statistics consumes a numpy array of the simulated portfolio results
    and intial_value representing the portfolio value intially and returns
    The Value at risk for levels alpha=0.95, alpha =99, and Expected Shortfall at 
    alpha = 0.98
    """
    e = numpy_results.sum(axis=0)
    print(e)
    e = e - intial_value
    ordered =  np.sort(e)
    
    VAR_95  = np.quantile(ordered, 0.05)
    VAR_99 = np.quantile(ordered, 0.01)
    VAR_98 = np.quantile(ordered, 0.02)
    # print(VAR_98)
    ES_98 = ordered[ordered <= VAR_98]
    ES_98 = np.average(ES_98)

    return([VAR_95, VAR_99, ES_98,intial_value])


def change_TM(matrix):
    # TODO Step 1 Handle the NR. a) Spread evenly b) proportional allocation
    #  c) To maintain monotonicity/ and increase Probability of Default
    # TODO Step 2  Handle the 0 PDs a) set them to minimum *take from somehwer else b) extrapolate (geometric or linear) 
    # c) Cubic / other advanced fitting methodlodies 
    # TODO Step 3 handling monotonicity 
    # 0) ignore it.
    # 1) Take from the NR. (Simple is adjust until monotone and allocate using Step 1 of NR)
    # 2) Take from the waiting stat ( stay at teh same rating level) and smooth it with NR
    #   (issues is that it over compnesates certain transitions)
    # 3) Twink it manually (hardest)
    
    # Step 1 Handling  PD (part 1)
    # We are setting minimum for PD to equal 0.01 or 0.01%
    matrix["Def"][matrix["Def"] == 0]  = 0.01
    # Now we take it away from NR
    matrix.loc["AAA", "NR"] -= 0.01
    
    # Step 2  Handling NR
    # a) Spread Proportional
    # Find rowsum from AAA  to Def 
    practice_np = np.array(matrix)
    da = practice_np[:,:-1] # refuses to use last column
    row_sum = np.sum(da,axis=1)
    #print(row_sum)
    # To divide component wise to the original matrix and find proportions with respect to row_sum
    da_new = da * np.reciprocal(row_sum[:,np.newaxis])
    #print(da_new) 
    # We are going to mutliple by NR column
    NR = practice_np[:,-1]
    #print(NR)

    adjust = da_new * NR[:,None]
    #print(adjust)

    matrix = matrix[["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C", "Def"]] + adjust
    
    
    ########  b) Evenly
    
    # states_n = practice_2.shape[1] - 1
    # # Take number of states, and subtract one because NR will be removed. Divide NR by number states, freely allocate.
    # temp_def =  practice_2[["NR"]] / states_n

    # d = practice_2[["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C", "Def"]]  + np.array(temp_def)
    # d.head()
    
    # Part 3: handling monotonoicty
    OG_AAA_BBB  = matrix.loc["AAA","BBB"] # store this information for later for adjusting diagonal
    AAA_BBB_new = (matrix.loc["AAA","A"]+  matrix.loc["AAA","BB"] )/2  
    matrix.loc["AAA","BBB"] = AAA_BBB_new
    Adjust_AAA_BBB = AAA_BBB_new - OG_AAA_BBB

    OG_AAA_B  = matrix.loc["AAA", "B"]
    AAA_B_new = (matrix.loc["AAA","BB"] + matrix.loc["AAA","CCC/C"] )/2
    matrix.loc["AAA","B"] = AAA_B_new
    Adjust_AAA_B  = AAA_B_new - OG_AAA_B

    OG_AA_BB  = matrix.loc["AA", "BB"]
    AA_BB_new = (matrix.loc["AA","BBB"] + matrix.loc["AA","B"]) /2
    matrix.loc["AA","BB"] = AA_BB_new
    Adjust_AA_BB  = AA_BB_new - OG_AA_BB

    matrix.loc["AAA","AAA"] -=  (Adjust_AAA_BBB + Adjust_AAA_B)
    matrix.loc["AA", "AA"] -= Adjust_AA_BB
    
    # Adjust for row_sum = 100
    missing = 100 - np.sum(matrix, axis =1)
    da_new_2 =  matrix * np.reciprocal(np.sum(matrix,axis=1)[:,np.newaxis])
    #print(da_new_2)
    #da = practice_np[:,:-1] # refuses to use last column
    row_sum_1 = np.sum(matrix,axis=1)
    #print(row_sum)
    # To divide component wise to the original matrix and find proportions with respect to row_sum
    #da_new_1 = np.product(matrix * np.reciprocal(row_sum_1 [:,np.newaxis]))
    adjustment=da_new * missing[:,None]
    #print(adjustment.head())
    matrix = matrix[["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C", "Def"]] + adjustment
    return(matrix)
    
