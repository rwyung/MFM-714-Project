import numpy as np
import pandas as pd
from scipy.linalg import cholesky
from scipy.stats import norm
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
    trans = os.path.join(data_url, "transition 2022.csv")
    
    # Read in dataf
    ratings = pd.read_excel(rate)
    portfolio =   pd.read_excel(port, index_col=0)
    yield_curve = pd.read_excel(yc,index_col=1,skiprows=[0], header= [1])
    transition =  pd.read_csv(trans,index_col=0)

    return( ratings, portfolio, yield_curve, transition)

def Correlation_Matrix(ratings,port):
    """
    Correlation_Matrix consume  a pd.DataFrame of portfolio positions
    and returns the associated correaltion matrix for each instrument

    Remark: 

    At the  moment correaltion matrix is filled on the diagnonal with 1s and
    0.36 The code will be changed at a later date
    """
    Comp_ = pd.unique(port.index)
    #print(Comp_)
    n_assets = len(Comp_) 
    
    Cor_Matrix = np.diag([1 for i in range(n_assets)])
    
    Cor_matrix_pd =  pd.DataFrame(Cor_Matrix, index = Comp_, columns=Comp_)
    # Slow Method 
    for comp1 in Cor_matrix_pd.index:
        correlation = 0
        for comp2 in Cor_matrix_pd.columns:
            if comp1 == comp2:
                correlation = 1 
            elif ratings.loc[comp1, "Industry"] == ratings.loc[comp2, "Industry"]:
                correlation = 0.65
            else: 
                correlation = 0.28
            Cor_matrix_pd.loc[comp1,comp2] = correlation
    
    
    return(Cor_matrix_pd)


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
    

### Step 2 Find points above spread.
def yc_spread(yc):
    """
    ~~~~~ PURPOSE ~~~~~
        yc_spread consumes a yield curve pd.DataFrame 'yc' 
        and returns the spread with respect to the Govt Yield
    ~~~~~ MAP ~~~~~~~~~    
        yc_spread: pd.DataFrame -> pd.DataFrame 
    """
    # print(yc.loc[:, yc.columns != 'Government'])
    #print(yc.loc[:,yc.columns != "Government"])
    Wo_Gov =  yc.loc[:, yc.columns != "Government"]
    Wo_Gov_1 = Wo_Gov.loc[:, Wo_Gov.columns != "Tenor"]
    output = Wo_Gov_1 - np.array(yc.loc[:, yc.columns == "Government"])
    return(output)


def p_above_spread(ratings, yc, port):
    
    """
    ~~~~~~~~ Purpose ~~~~~~~~
    p_above_spread consumes a pd.DataFrame 'ratings' representing the ratings  and pd.DataFrame 'yc'representing the yield curve
    and returns a pd.Dataframe with all the companies and respective spreads, and idiosyncratic bps above rating spread.
    ~~~~~~~~ Function ~~~~~~~
    p_above_spread:  pd.DataFrame, pd.DataFrame =>  pd.DataFrame
    
    """

    
    
    # TODO Step 1 is join tables with respective yield and year.  Should include table with just the year to maturity and subtract gov yields
    #       Step 2 Find the spread of the rating with respect to government yields 
    #       Step 3 Find the difference between spreads and viola you have idiosyncratic spread adding factors 
    
    
    ########## Step 1 ###########
    ### Key steps 
    ### Define the price of the bond and find yield that it is currently trading for.
    ### Take the yield found and subtract by governent yield to find bond spread .
    ### Not in particular order, find the spread with respect to the yield of the rating of the bond with respect to  Gov.t
    
    
    pass



def time_maturity(portfolio,start_date = date(2021,1,1)):
    """
        time_maturity consumes a pd.DataFrame representing the portfolio and a datetime object called 'start_date'
        and returns a portfolio pd.DataFrame with 'T2M' column representing the "Time to Maturity"

        time_maturity: pd.DataFrame, datetime.date -> pd.DataFrame
    """

    portfolio["T2M"] = portfolio["Maturity Date"].apply(lambda x: (relativedelta(x,start_date).years +(relativedelta(x,start_date).months/12)))
    portfolio["T2M"] = np.where(portfolio["T2M"] <= 0, portfolio["T2M"] + 1, portfolio["T2M"]) # checks for negative dates,  and adds a year.
    return(portfolio)

    

def makehams_formula(coup, rate, time_remaining, freq, Notional):
    ye = np.floor(time_remaining) 
    adj = time_remaining - ye
    t2 = ye.copy()
    # print(adj)
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
    Dirty = (coup/rate * Notional *(1 - D)  + Notional * D)
    # Dirty_price represents the Dirty Price of the bond
    P = Dirty * (1 + rate/2)**adjust  -   Notional*coup*adjust
    #return(pd.from_dict({"Clean Price":P,"Dirty: Price" : Dirty },dtype= np.float64)) 
    return(P)

def CDS_pricer(status,Instrument,recovery,Notional):
    if Instrument == "CDS":
        if  status == "Default":
            return(Notional * recovery)
        else:
            return(0)

def  lookup_yield_curve(port,ref):
    port["Grade rate"] = ref.lookup(port["T2M"], port["Issuer Rating"])
    
def adjust_yield_curve(port,yc):
    col_names = yc.columns 
    col_names.remove("government","In years")
    ret = port.copy()
    time_now = date(2021,1,1)
    #ret["status"]= ret["Instrument type (Bond or CDS)"].apply(lambda x: 1 if x == "Bond" else 0 )
    # ret["Years_Remaining"] =  ret["Maturity_Date"].apply(lambda x: (relativedelta(x,now_w).years +(relativedelta(x,now_w).months/12))) - 1
    # ret["Years_remaining"] = np.wherse(ret["Years Remaining"] <= 0, ret["Years Remaining"] + 1, ret["Years Remaining"])
    ret = time_maturity(ret, time_now)
    
    #result = pd.DataFrame(columns=col_names)
    
    ### Adjusts yield _cur
    df =  yc.copy()
    for i in ret["T2M"]:
        if i not in yc.index:
            df.loc[i] = np.nan
   
    df.sort_index(axis= 0,inplace = True)
    df = df.interpolate(method="linear",axis=0) #new yc with times in between time to maturities.
    df.fillna(axis= 0, method="bfill", inplace = True) 
    # back fills rates for short term rates
    df = df[df.index.notnull()] # removes na indexes
    return(df)

def CDS_Price(coup, spread,t2m,ye,freq):
    D = (1/(1+ ye/freq)**(freq*t2m))
    result = (coup - spread )/ ye *  (1-D)
    return(result)




def CDS_Price(N, coup, spread,t2m,ye,freq):
    D = (1/(1+ ye/freq)**(freq*t2m))
    result =  N *  (spread - coup )/ ye *  (1-D)
    return(result)
def beta_recovery(ratings,alpha=1.62, beta=1.86):
    num_of_companies = ratings.shape[0]
    recoveries=np.random.beta(alpha,beta, (num_of_companies, 1))
    ratings["Recovery"] = recoveries
    return(ratings)

# I want to split the data to two different portfolios 

def one_year_forward(port,transition,yc):
    # Comment you want to use the yc for one year later: We will get new years 
    comp_uniq = np.unique(port.index)
    grades = np.unique(transition.columns)
    result = pd.DataFrame(index = comp_uniq,columns= grades)
    temp = port.copy()
    #TODO LOOKUP Table and return rates add the idiosyncratic adjustment and bam.
    # First we must reduce  the T2M by one year
    temp["T2M_1yr"] = temp["T2M"] - 1 
    # If maturities after 1 year are less than 0 set to 1
    temp["T2M_1yr"] = np.where(temp["T2M_1yr"] <0 , 1 , temp["T2M_1yr"])
    #temp2["Grade rate"] = temp.lookup(temp2["T2M"], temp2["Issuer Rating"]
        # temp_grades  = grades.tolist()
    # temp_grades.remove("D") # remove default from considerations
    desired_col = ["Unique_id", "Instrument type (Bond or CDS)", "T2M_1yr","Coupon","ISpread", "Coupons per year",
     "Notional","Recovery", "Yield"]
    
    grades_list = grades.tolist()
    #grades_list.remove("D")

    result = temp[desired_col]
    result.reset_index(inplace=True)
    temp_v  = pd.DataFrame({"New Ratings": grades_list})
    result_f = pd.merge(result, temp_v, how="cross")
    result_f.set_index("Name", inplace=True)
    yc["D"] = 0.5
    result_f["Sim Yields"] = yc.lookup(result_f["T2M_1yr"], result_f["New Ratings"])
    result_f["New Yields"]  = (result_f["ISpread"].values + result_f["Sim Yields"].T).T
    
    
    result_f["Sim Prices"] = np.where(result_f["Instrument type (Bond or CDS)"] == "Bond", utils.makehams_formula(
        result_f.Coupon, result_f["New Yields"], result_f["T2M_1yr"],result_f["Coupons per year"],result_f["Notional"]
    ), CDS_Price(result_f["Notional"],result_f["Coupon"],result_f["New Yields"],result_f["T2M_1yr"],result_f["New Yields"],result_f["Coupons per year"]))
    Step3 = result_f.copy()
    Step3["Sim Prices"] = np.where((Step3["New Ratings"] == "D") & (Step3["Instrument type (Bond or CDS)"] == "Bond"), calculate_recovery(Step3["Recovery"], Step3["Notional"]), Step3["Sim Prices"])
    Step3["Sim Prices"] =  np.where((Step3["New Ratings"] == "D") & (Step3["Instrument type (Bond or CDS)"] == "CDS"), calculate_recovery(1-Step3["Recovery"], Step3["Notional"]),Step3["Sim Prices"])


    #Step3[(Step3["New Ratings"] == "D") & (Step3["Instrument type (Bond or CDS)"] == "Bond")]["Sim Prices"] = calculate_recovery(Step3["Recovery"], Step3["Notional"] )    
    #Step3[(Step3["New Ratings"] == "D") & (Step3["Instrument type (Bond or CDS)"] == "Bond")]["Sim Prices"] = calculate_recovery((1-Step3["Recovery"]), Step3["Notional"] )
    
    return(Step3)

def look_for_rating(x, cdftrans, initial_rating):
    d= cdftrans.loc[[initial_rating]]
    look = d.T
    rating=look[look[initial_rating].le(x)].index[-1]
    return(rating)