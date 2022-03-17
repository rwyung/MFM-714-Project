
import utils
import numpy as np
import pandas as pd


ratings, portfolio, yield_curve, transition = utils.upload_data()

port_Cor = utils.Correlation_Matrix(ratings)

P_mrkt  = utils.mkt_value(portfolio)

Total = P_mrkt["Market Value"].sum()

Forward_scenarios = utils.one_year_forward(P_mrkt,transition)

hash_table = Forward_scenarios.to_dict('records')

MC_ratings_sim = utils.ratings_sim(P_mrkt, ratings, N=2000)
MC_ratings_sim = MC_ratings_sim.reset_index().set_index(["Name","Instrument type (Bond or CDS)"])
Prices_MC = utils.pricing_matrix(MC_ratings_sim,hash_table)

market_values = utils.Mc_price(Prices_MC,np.array(P_mrkt["Notional"]))
port_sim_values = market_values
stats = utils.Calculate_Statistics(market_values,Total)

if __name__ == "__main__":
    print("-------------Available Data (Part 1)------------ ")
    print(ratings.head())
    print(portfolio.head())
    print(yield_curve.head())
    print(transition.head())
    print("-------------Step 2  ------------ ")
    print(P_mrkt["Market Value"].head())
    print("-------------Step 3  ------------ ")
    print(Forward_scenarios.head())
    print("-------------Step 4  ------------ ")
    print(MC_ratings_sim.head())
    print("-------------Step 5  ------------ ")
    print(Prices_MC)
    print(market_values)
    print("-------------Step 6  ------------ ")
    print("VAR 95: {} , VAR 99: {} , ES_98: {}, Initial Port Value: {}".format(*stats))