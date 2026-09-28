"""Regression checks for the two diagnosed numerical/provenance defects."""
import json
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from v9_7_numerics import gap_cdf,speed_cdf,moment_nodes,gap_cdf_gpu


class IntegrationRepairTests(unittest.TestCase):
    def setUp(self):
        self.pred={k:np.array([v]) for k,v in dict(logQ=1.2,logita=-.2,logJ=-1.,logS=.1).items()}
        qa=np.array([[.9**2,.25],[.25,1.1**2]]);js=np.array([[1.2**2,.3],[.3,.5**2]])
        self.cal=dict(std_logQ=.9,std_logita=1.1,std_logJ=1.2,std_logS=.5,a_const=.48,covariance_QA=qa.tolist(),cholesky_QA=np.linalg.cholesky(qa).tolist(),covariance_JS=js.tolist(),cholesky_JS=np.linalg.cholesky(js).tolist())

    def test_js_analytic_marginal_and_old_three_atom_error(self):
        g=np.geomspace(.001,20,151);u=np.array([5.])
        expected=ndtr((np.log(g)+1)/1.2)
        for model in ('Prob_JS_ind','Prob_JS_joint'):
            actual=gap_cdf(self.pred,u,self.cal,model,g)[0]
            np.testing.assert_allclose(actual,expected,atol=1e-14)
        _,atoms,_,weights=moment_nodes(self.pred,u,self.cal,'Prob_JS_joint',3)
        old=((atoms[:,0,None]<=g)*weights[:,None]).sum(axis=0)
        self.assertGreater(float(abs(old-expected).max()),.2)

    def test_conditional_cdf_against_adaptive_integral(self):
        for model in ('QA_dyn','QA_m3'):
            for u in (0.,.2,5.,20.):
                for j in (.01,.3,2.):
                    cov=.25 if model=='QA_m3' else 0.
                    sd=np.sqrt(.9**2-cov**2/1.1**2)
                    def integrand(z):
                        loga=-np.logaddexp(0,.2-1.1*z)
                        mean=1.2+cov/1.1*z
                        return ndtr((np.log(j*(2*u+j))-loga-mean)/sd)*np.exp(-z*z/2)/np.sqrt(2*np.pi)
                    expected=quad(integrand,-11,11,epsabs=1e-12)[0]
                    actual=gap_cdf(self.pred,np.array([u]),self.cal,model,np.array([0.,j]),64)[0,1]
                    self.assertAlmostEqual(actual,expected,places=9)

    def test_constant_inverse_and_cdf_boundaries(self):
        u=np.array([5.]);g=np.r_[0.,np.geomspace(.001,100,100)]
        actual=gap_cdf(self.pred,u,self.cal,'QA_const',g)[0]
        expected=ndtr((np.log(g[1:]*(10+g[1:])/.48)-1.2)/.9)
        np.testing.assert_allclose(actual[1:],expected,atol=1e-14)
        self.assertEqual(actual[0],0.)
        self.assertTrue(np.all(np.diff(actual)>=0))

    def test_cuda_matches_float64_reference(self):
        import torch
        if not torch.cuda.is_available():self.skipTest('CUDA unavailable; CPU is the reference')
        g=np.r_[np.linspace(0,30,241),np.geomspace(31,4096,60)]
        for model in ('QA_dyn','QA_m3','QA_const','Prob_JS_ind','Prob_JS_joint'):
            cpu,_=speed_cdf(self.pred,np.array([5.]),self.cal,model,g,24,device='cpu')
            gpu,_=speed_cdf(self.pred,np.array([5.]),self.cal,model,g,24,device='cuda')
            np.testing.assert_allclose(cpu,gpu,atol=2e-6,rtol=0)
            a=gap_cdf(self.pred,np.array([5.]),self.cal,model,g)
            b=gap_cdf_gpu(self.pred,np.array([5.]),self.cal,model,g)
            np.testing.assert_allclose(a,b,atol=3e-7,rtol=0)

    def test_invalid_covariance_rejected(self):
        self.cal['covariance_QA']=[[.81,2],[2,1.21]]
        with self.assertRaises(ValueError):gap_cdf(self.pred,np.array([5.]),self.cal,'QA_m3',np.array([0.,1.]))


if __name__=='__main__':unittest.main(verbosity=2)
