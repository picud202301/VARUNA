

from __future__ import annotations

def loadProblem(problem_name, problem_rng):
    solvers_parameters = {}
    solvers_parameters['integration'] = 'euler'
    if problem_name == 'zermelo':
        # ---------------------- Configure solvers ----------------------
        from problems.zermelo.solvers.SolverZermeloAnalytic import SolverZermeloAnalytic
        from problems.zermelo.solvers.SolverZermeloIpopt import SolverZermeloIpopt
        from problems.zermelo.solvers.SolverZermeloAStar import SolverZermeloAStar
        from problems.zermelo.solvers.SolverZermeloPSO import SolverZermeloPSO
        from problems.zermelo.ProblemZermelo import ProblemZermelo
        solvers_configuration = {
            'analytic':   {'class': SolverZermeloAnalytic, 'active': True, 'parameters':{'color':'red',  'library': 'np'}},
            'astar':      {'class': SolverZermeloAStar,     'active': True, 'parameters':{'color':'blue',  'library': 'np'}},
            'pso':        {'class': SolverZermeloPSO,       'active': True, 'parameters':{'color':'black',  'library': 'np'}},
            'ipopt':      {'class': SolverZermeloIpopt,    'active': True, 'parameters':{'color':'green',  'library': 'pyo'}},
        }
        problem = ProblemZermelo(rng = problem_rng)

    return problem, solvers_configuration, solvers_parameters

    

    
    
