import torch
import numpy as np

class NNSurrogate:
    def __init__(
        self,
        model,
        CL_mean, CL_std,
        dlogCD_mean, dlogCD_std,
        LD_mean, LD_std,
        xgb_CL,
        xgb_CD,
        device=None
    ):
        self.model = model.eval()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.CL_mean = CL_mean
        self.CL_std = CL_std
        self.dlogCD_mean = dlogCD_mean
        self.dlogCD_std = dlogCD_std
        self.LD_mean = LD_mean
        self.LD_std = LD_std

        self.xgb_CL = xgb_CL
        self.xgb_CD = xgb_CD

    def predict_sweep(self, geom_dict, Re, alpha_sweep):
        n = len(alpha_sweep)
        X_base = np.zeros((n, 7), dtype=np.float32)
        X_base[:, 0] = geom_dict['Aspect']
        X_base[:, 1] = geom_dict['Taper']
        X_base[:, 2] = geom_dict['Sweep']
        X_base[:, 3] = geom_dict['Dihedral']
        X_base[:, 4] = geom_dict['Twist']
        X_base[:, 5] = alpha_sweep
        X_base[:, 6] = np.log10(Re)

        CL_LF = self.xgb_CL.predict(X_base).astype(np.float32)
        CD_LF = np.clip(self.xgb_CD.predict(X_base), 1e-10, None).astype(np.float32)

        X = np.column_stack([X_base, CL_LF, CD_LF]).astype(np.float32)
        X_tensor = torch.from_numpy(X).to(self.device)

        with torch.no_grad():
            cl_pred, dlogcd_pred, ld_pred = self.model(X_tensor)

        cl_pred = cl_pred.cpu().numpy() * self.CL_std + self.CL_mean
        ld_pred = ld_pred.cpu().numpy() * self.LD_std + self.LD_mean
        dlogcd_pred = dlogcd_pred.cpu().numpy() * self.dlogCD_std + self.dlogCD_mean
        cd_pred = np.exp(np.log(CD_LF) + dlogcd_pred.flatten())

        return {
            'CL': cl_pred.flatten(),
            'CD': cd_pred.flatten(),
            'LD': ld_pred.flatten()
        }


