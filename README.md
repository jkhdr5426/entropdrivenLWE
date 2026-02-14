EO-LWE experimental pipeline
===========================

Files:
- requirements.txt
- lwe_core.py
- transforms.py
- sample_generator.py
- run_grid_experiment.py
- attack_proxies.py
- lattice_prep.py
- analyze_and_plot.py
- (honest_test.py is a helper placeholder)

Install:
$ python -m venv venv
$ source venv/bin/activate    # Windows: venv\\Scripts\\activate
$ pip install -r requirements.txt

Quick smoke test (small, runs in seconds):
$ python run_grid_experiment.py --transform quadratic --n 16 --q 3329 --sigma 2.0 --trials 200 --alpha_range 8 --out_dir results_quadratic_n16

Check results:
$ ls results_quadratic_n16
- summary_quadratic_n16.csv
- samples_quadratic_n16.csv
- lin_quadratic_n16.npz   # linearized matrix for lattice analysis

Analyze & plot:
$ python analyze_and_plot.py --results_dir results_quadratic_n16

Repeat for multiple cells, e.g.:
$ python run_grid_experiment.py --transform quadratic --n 32 --trials 300 --out_dir results_quad_n32
$ python run_grid_experiment.py --transform cyclotomic_toy --n 32 --trials 300 --out_dir results_cyc_n32
$ python run_grid_experiment.py --transform matrix_lift --n 32 --trials 300 --out_dir results_mat_n32

Lattice (optional):
- The file lin_quadratic_n16.npz contains arrays A and b of the linearized system.
- To create an fplll basis and run BKZ you need to construct a q-ary lattice basis from A and b (not provided as a one-liner here).
- If you install fpylll you can adapt lattice_prep.py to produce fplll's basis format.

Notes:
- Do NOT publish the secret s. The code writes sample files with public data only.
- Small-n experiments are for prototyping: do not claim asymptotic security.
- For heavy BKZ runs use a cloud server with many CPUs and RAM.
