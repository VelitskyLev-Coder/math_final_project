import sympy as sp

n_a = sp.symbols('N_a')
n_b = sp.symbols('N_b')
r_a = sp.symbols('r_a')
r_b = sp.symbols('r_b')
m = sp.symbols('m')
e_a = sp.symbols('e_a')
e_b = sp.symbols('e_b')

f_a = r_a * n_a / (1 + n_a)
f_b = r_b * n_b / (1 + n_b)

subject1 = n_a - ((1 - m) * f_a.subs(n_a, n_a) + m * f_b.subs(n_b, n_b))*(1-e_a)
subject2 = n_b - (m * f_a.subs(n_a, n_a) + (1 - m) * f_b.subs(n_b, n_b))*(1-e_b)

g = f_a.subs(n_a, n_a) + f_b.subs(n_b, n_b) - (n_a+n_b)

lambda1 = sp.symbols('lambda1')
lambda2 = sp.symbols('lambda2')

L = g + lambda1*subject1 + lambda2*subject2

L_n_a = sp.simplify(sp.diff(L, n_a))
L_n_b = sp.simplify(sp.diff(L, n_b))
L_e_a = sp.simplify(sp.diff(L, e_a))
L_e_b = sp.simplify(sp.diff(L, e_b))
L_lambda1 = sp.simplify(sp.diff(L, lambda1))
L_lambda2 = sp.simplify(sp.diff(L, lambda2))

L_n_a *= (n_a+1)**2
L_n_b *= (n_b+1)**2
L_e_a *= (n_a+1)+(n_b+1)
L_e_b *= (n_a+1)*(n_b+1)
L_lambda1 *= (n_a+1)*(n_b+1)
L_lambda2 *= (n_a+1)*(n_b+1)

L_n_a = sp.simplify(L_n_a)
L_n_b = sp.simplify(L_n_b)
L_e_a = sp.simplify(L_e_a)
L_e_b = sp.simplify(L_e_b)
L_lambda1 = sp.simplify(L_lambda1)
L_lambda2 = sp.simplify(L_lambda2)

#print(sp.latex(L))
# print(f'{sp.latex(L_n_a)}=0\\\\\\\\')
# print(f'{sp.latex(L_n_b)}=0\\\\\\\\')
# print(f'{sp.latex(L_e_a)}=0\\\\\\\\')
# print(f'{sp.latex(L_e_b)}=0\\\\\\\\')
# print(f'{sp.latex(L_lambda1)}=0\\\\\\\\')
# print(f'{sp.latex(L_lambda2)}=0\\\\\\\\')

n_a_val = sp.solve(L_n_a.subs(lambda1, 0).subs(lambda2, 0), n_a)[1]
n_b_val = sp.solve(L_n_b.subs(lambda1, 0).subs(lambda2, 0), n_b)[1]
print(f'n_a={sp.latex(n_a_val)}')
print(f'n_b={sp.latex(n_b_val)}')
e_a_val = sp.simplify(sp.solve(L_lambda1.subs(n_a, n_a_val).subs(n_b, n_b_val), e_a)[0])
e_b_val = sp.simplify(sp.solve(L_lambda2.subs(n_a, n_a_val).subs(n_b, n_b_val), e_b)[0])

print(f'e_a_val={sp.latex(e_a_val)}')
print(f'e_b_val={sp.latex(e_b_val)}')