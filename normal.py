from scipy.stats import norm
from scipy.stats import multivariate_normal
import math


x1 = [2, 4]
x2 = [-1, -4]
x3 = [-1, 2]
x4 = [4, 0]

means1 = [2, 4]
means2 = [-1, -4]

covs1 = [[1,0],[0,1]]
covs2 = [[2,0],[0,2]]

prior1 = 0.7
prior2 = 0.3

def getNormal(means, covs, points):
    return multivariate_normal(means, covs).pdf(points)

def getJointProbability(prior, normal):
    return prior * normal

def getNormalizedProbability(jointXC1, jointXC2):
    return jointXC1 / (jointXC1 + jointXC2)

x = x4

print("-"*50+"\n")


normalXC1 = getNormal(means1,covs1,x)
print("Normal C1 -> " + str(normalXC1) +"\n")

JointXC1 = getJointProbability(prior1,normalXC1)
print("Joined Probability C1 -> " + str(JointXC1)+"\n")

normalXC2 = getNormal(means2, covs2, x)
print("Normal C2 -> " + str(normalXC2)+"\n")

JointXC2 = getJointProbability(prior2,normalXC2)
print("Joined Probability C2 -> " + str(JointXC2)+"\n")

NormalizedProbC1 = getNormalizedProbability(JointXC1, JointXC2)
print("Normalized Probability C1 -> " + str(NormalizedProbC1)+"\n")

NormalizedProbC2 = getNormalizedProbability(JointXC2, JointXC1)
print("Normalized Probability C2 -> " + str(NormalizedProbC2)+"\n")


print("-"*50+"\n")

