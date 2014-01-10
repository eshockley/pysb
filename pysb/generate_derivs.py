import sympy
from pysb.bng import generate_equations

def gen_jacobian(model):
    if model.jacobian:
        return
    generate_equations(model)
    jacmatrix = sympy.S([range(len(model.odes)) for i in range(len(model.odes))])
    for i in range(len(model.odes)):
        for x in range(len(model.odes)):
            species = 's'+str(x)
            jacmatrix[i][x] = sympy.diff(model.odes[i], species)
    model.jacobian = jacmatrix

def gen_hessian(model):
    if model.hessian:
        return
    generate_equations(model)
    gen_jacobian(model)
    hmatrix = sympy.S([range(len(model.odes)) for i in range(len(model.odes))])
    for i in range(len(model.odes)):
        for x in range(len(model.odes)):
            species = 's'+str(x)
            hmatrix[i][x] = sympy.diff(model.jacobian[i][x], species)
    model.hessian = hmatrix

def gen_jacobian_params(model):
    if model.jacobian_params:
        return
    generate_equations(model)
    pnames = [p.name for p in model.parameters_rules()]
    jacmatrix = sympy.S([range(len(model.parameters_rules())) for i in range(len(model.odes))])
    for i in range(len(model.odes)):
        for x, name in zip(range(len(model.parameters_rules())), pnames):
            jacmatrix[i][x] = sympy.diff(model.odes[i], name)
    model.jacobian_params = jacmatrix

def gen_hessian_params(model):
    if model.hessian_params:
        return
    generate_equations(model)
    gen_jacobian_params(model)
    pnames = [p.name for p in model.parameters_rules()]
    hmatrix = sympy.S([range(len(model.parameters_rules())) for i in range(len(model.odes))])
    for i in range(len(model.odes)):
        for x, name in zip(range(len(model.parameters_rules())), pnames):
            hmatrix[i][x] = sympy.diff(model.jacobian_params[i][x], name)
    model.hessian_params = h
