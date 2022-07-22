def stokes_vector(P,S):

    """ Stokes vector out of P and S polarization states

            : param P (complex 1xn array): p-polarization state
            : param S (complex 1xn array): s-polarization state

            : return [I,Q,U,V] (real 4xn np.array): Stokes vector along z

    """

    import numpy as np

    I = np.abs(P)**2+np.abs(S)**2
    Q = np.abs(P)**2-np.abs(S)**2
    product = P*np.conj(S)
    U = 2*product.real
    V = 2*product.imag

    return np.array([I,Q,U,V])
