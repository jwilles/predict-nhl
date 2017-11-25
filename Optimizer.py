
from gurobipy import *
import numpy as np
import pandas as pd
from Predictor import ownershipPredict
from scipy import sparse

class optimize(object):
    def A_opt(self, df):
        return np.transpose(np.array(df[['D','K','QB','RB','TE','WR','salary']].copy()))


    def c_opt(self, df):
        return np.array(df.proj_points)

    def c2_opt(self, df):
        return np.array(df['Predicted Ownership'])

    def b_opt(self):
        return np.array([1, 1, 1, 2, 1, 3, 60000])


    def opt_lineup(self, df,objWeights):
        cols = len(df)
        rows = 7
        output = []
        A = self.A_opt(df)
        c_points = self.c_opt(df)
        c_ownership = self.c2_opt(df)
        rhs = self.b_opt()

        model = Model()
        model.setParam('OutputFlag', False)
        sense = 6 * [GRB.EQUAL] + [GRB.LESS_EQUAL]
        vars = []
        obj_points = LinExpr()
        obj_ownership = LinExpr()

        #Populate objective functions
        for j in range(cols):
            vars.append(model.addVar(vtype=GRB.BINARY)) #, obj=c[j]
            obj_points += c_points[j] * vars[j]
            obj_ownership += c_ownership[j] * vars[j]

        #Populate constraint matrix
        for i in range(rows):
            expr = LinExpr()
            for j in range(cols):
                if A[i][j] != 0:
                    expr += A[i][j] * vars[j]
            model.addConstr(expr, sense[i], rhs[i])
        model.update()

        obj_total = objWeights[0] * obj_points - objWeights[1] * obj_ownership

        model.setObjective(obj_total, GRB.MAXIMIZE)
        model.update()

        model.optimize()
        for v in model.getVars():
            output.append(v.x)

        return np.array(output).ravel()

    def solve_lineup(self, objWeights):
        lineup = pd.DataFrame([])
        df = pd.read_csv(os.path.dirname(os.path.realpath(__file__)) + '/temp/our_projections.csv')
        sol = self.opt_lineup(df, objWeights)
        locations = np.array(np.where(sol==1)).ravel()
        for x in locations: lineup = lineup.append(df[['play','Position','salary','proj_points','Predicted Ownership']].loc[[x]])
        lineup.columns = ['Player', 'Position', 'Salary', 'Projected Points', 'Projected Ownership']
        totals = ['Total', 'N/A', sum(lineup.Salary), sum(lineup['Projected Points']), sum(lineup['Projected Ownership']) / 9]
        totals = pd.DataFrame([totals], columns=['Player', 'Position', 'Salary', 'Projected Points', 'Projected Ownership'])
        return lineup.append(totals)


