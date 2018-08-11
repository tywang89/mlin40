from pgmpy.models import MarkovModel
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.inference import VariableElimination

phi_1 = DiscreteFactor(['A', 'B'], [2, 2], [30, 5, 1, 10])
phi_2 = DiscreteFactor(['B', 'C'], [2, 2], [100, 1, 1, 100])
phi_3 = DiscreteFactor(['C', 'D'], [2, 2], [1, 100, 100, 1])
phi_4 = DiscreteFactor(['D', 'A'], [2, 2], [100, 1, 1, 100])

model = MarkovModel([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])
model.add_factors(phi_1, phi_2, phi_3, phi_4)
phi = phi_1 * phi_2 * phi_3 * phi_4
Z = model.get_partition_function()
normalized = phi.values / Z

print(normalized)
