"""Recompute window diagnostics from saved BGK histories without rerunning physics."""
from pathlib import Path
import argparse,json
import numpy as np
from bgk_itg_saturation import analyze,plot


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directories',nargs='+');ap.add_argument('--start',type=float)
    a=ap.parse_args();summary=[]
    for dirname in a.directories:
        out=Path(dirname);params=json.loads((out/'parameters.json').read_text());params.setdefault('model','local')
        data=dict(np.load(out/'history.npz'));start=params['average_start'] if a.start is None else a.start
        report,spectra=analyze(data,start,params['at']);report['parameters']=params
        report['parameters']['average_start']=start
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        np.savez_compressed(out/'mean_spectra.npz',**spectra)
        ky=.3*np.fft.fftfreq(params['ny'])*params['ny'];plot(out,data,report,spectra,ky)
        summary.append(dict(path=dirname,model=params['model'],nu=params['nu'],nx=params['nx'],ny=params['ny'],
            start=start,end=float(data['t'][-1]),Q=report['mean_heat_flux'],window_error=report['heat_block_standard_error'],
            power_imbalance=report['mean_power_relative_imbalance'],budget_error=report['budget_relative_max'],
            heat_change=report['heat_half_relative_change'],W_change=report['W_half_relative_change'],
            spectrum_change=max(report['spectral_half_relative_changes'].values()),
            velocity_edge=report['velocity_last_order_energy_fraction'],radial_edge=report['radial_last_three_energy_fraction'],
            passed=report['stationarity_screen_pass']))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
