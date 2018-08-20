from pgmpy.models import BayesianModel, JunctionTree
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import BeliefPropagation

model = BayesianModel()
model.add_nodes_from(['family_out','bowel_problem', 'light_on','dog_out','hear_bark'])
model.add_edge('family_out', 'light_on')
model.add_edge('family_out', 'dog_out')
model.add_edge('bowel_problem', 'dog_out')
model.add_edge('dog_out', 'hear_bark')

cpd_fo = TabularCPD(variable='family_out', variable_card=2, values=[[0.15], [0.85]])
cpd_bp = TabularCPD(variable='bowel_problem', variable_card=2, values=[[0.01], [0.99]])
cpd_do = TabularCPD(variable='dog_out', variable_card=2, 
                   values=[[0.99, 0.9, 0.97, 0.3],[0.01, 0.1, 0.03, 0.7]],
                  evidence=['family_out', 'bowel_problem'], evidence_card=[2, 2])
cpd_lo = TabularCPD(variable='light_on', variable_card=2,
                   values=[[0.6, 0.05],[0.4, 0.95]], evidence=['family_out'], evidence_card=[2])
cpd_hb = TabularCPD(variable='hear_bark', variable_card=2,
                   values=[[0.7, 0.01],[0.3, 0.99]], evidence=['dog_out'], evidence_card=[2])

#integrity checking
model.add_cpds(cpd_fo, cpd_bp, cpd_do, cpd_lo, cpd_hb)
model.check_model()

junction_tree = model.to_junction_tree()
print(junction_tree.nodes())

infer_bp = BeliefPropagation(junction_tree)
print(infer_bp.query(['family_out'], evidence={'light_on': 0, 'hear_bark': 1}) ['family_out'])


