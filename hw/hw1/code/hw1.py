# For problem 2.1.b
x = [[1, 1], [4, 4], [1, -2], [4, 1]]
d = [1, 1, 0, 0]
curr_w, new_w = [0, 0], [0, 0]
curr_b, new_b = 0, 0
r = 0.1
vals_changed = True


def isclose(a, b, rtol=1e-5):
    return abs(a - b) < abs(b * rtol)


def allclose(a, b, rtol=1e-5):
    for a_i, b_i in zip(a, b):
        if not isclose(a_i, b_i, rtol):
            return False
    return True


print(f"learning_rate={str(r)}")
while vals_changed:    
    print(f"{curr_w=} {curr_b=}")
    print("Performing weight update")
    vals_changed = False

    y = []
    for j in range(len(x)):    
        # Calculate current output
        y.append(int(sum([curr_w[i] * x[j][i] for i in range(len(x[j]))], start=curr_b) > 0))

    for d_j, x_j, y_j in zip(d, x, y):
        # Weight update
        for i in range(len(curr_w)):
            new_w[i] += r*(d_j - y_j)*x_j[i]
        new_b += r*(d_j - y_j)
        
    vals_changed |= not allclose(curr_w, new_w)
    vals_changed |= not isclose(curr_b, new_b)
    curr_w, curr_b = new_w, new_b

print(curr_w, curr_b)
