import random

def generateInstance(n, k, m):
    
    vars = [chr(i + 65) for i in range(n)]
    problem = []
    clause = []

    
    for i in range(k * m):
        
        x = random.choice(vars)
        vars.remove(x)
        clause.append(x)

        
        if random.random() < 0.5:
            problem.append(f"~{x}")
        else:
            problem.append(x)

       
        if (i + 1) % k == 0:
            vars.extend(clause)
            clause.clear()

    
        if (i + 1) % k == 0 and i != (k * m - 1):
            problem.append(") and (")
        elif i != (k * m - 1):
            problem.append(" or ")


    return "((" + "".join(problem) + "))"


for i in range(10):
    print(f"Problem {i + 1}: {generateInstance(12, 3, 4)}")
