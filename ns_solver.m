stageH2F_T2_full_bspf_projected_rhs_rk4_weakfilter_RHSK_L1_integrated_v2_default_core_encoding_fixed_v2
function results = stageH2F_T2_full_bspf_projected_rhs_rk4_weakfilter_RHSK_L1_integrated_v2_default_core_encoding_fixed_v2(opts)
%STAGEH2F_T2_FULL_BSPF_PROJECTED_RHS_RK4_WEAKFILTER_RHSK_L1_INTEGRATED_V2
% -------------------------------------------------------------------------
% H2F validation: full-BSPF projected-RHS RK4 with weak RHS/K-only BSPF/Fourier filtering 2D incompressible
% Navier-Stokes MMS solver on Omega=[0,1]x[0,1].
%
% This is the first multi-step test where the velocity state is stored and
% advanced as a complete BSPF representation:
%
%   F = F_BB + F_BF + F_FB + F_per.
%
% At every RK4 stage:
%   1. u,v and their derivatives are evaluated from BSPF coefficients;
%   2. the non-pressure RHS is built using BSPF derivatives;
%   3. RHS components are re-represented in complete BSPF form;
%   4. div(R) and boundary R.n are computed from BSPF RHS coefficients;
%   5. BSPF-KKT Neumann Poisson projection gives K = P(R);
%   6. K is re-represented in complete BSPF form.
%
% RK4 updates are performed by linear combinations of BSPF coefficient
% objects, not by updating finite-difference grid values.
%
% Optional cleanup projection is applied at the end of each time step and
% the cleaned velocity is re-represented in complete BSPF form.
%
% Usage:
%   results = stageH2F_T2_full_bspf_projected_rhs_rk4_weakfilter_RHSK_L1_integrated_v2_default_core();
%
% Optional:
%   opts = struct();
%   opts.Nx = 100; opts.Ny = 100;
%   opts.dt = 0.02; opts.nSteps = 10;
%   results = stageH2F_T2_full_bspf_projected_rhs_rk4_weakfilter_RHSK_L1_integrated_v2_default_core(opts);
% -------------------------------------------------------------------------

if nargin < 1 || isempty(opts), opts = struct(); end
opts = cmp_set_default(opts, 'Lx', 1.0);
opts = cmp_set_default(opts, 'Ly', 1.0);
opts = cmp_set_default(opts, 'Nx', 100);
opts = cmp_set_default(opts, 'Ny', 100);
opts = cmp_set_default(opts, 'degB', 6);
opts = cmp_set_default(opts, 'nbasis', 18);
opts = cmp_set_default(opts, 'kmax', 5);
opts = cmp_set_default(opts, 'r', 3);
opts = cmp_set_default(opts, 'endpointDerivativeMethod', 'local-poly-qr');  % 'fd' or 'local-poly-qr'
opts = cmp_set_default(opts, 'endpointDerivativeRadius', opts.r + 1);
opts = cmp_set_default(opts, 'endpointDerivativeDegree', opts.kmax + 2);
opts = cmp_set_default(opts, 'requireBspfOnlyDerivatives', true);
opts = cmp_set_default(opts, 'allowEndpointDerivativeStencilInBspfSplit', true);
opts = cmp_set_default(opts, 'useNoSlipVelocityEnvelope', false);
opts = cmp_set_default(opts, 'useBoundaryConstrainedDecomposition', false);
opts = cmp_set_default(opts, 'useKktEndpointBoundaryConstraints', true);
opts = cmp_set_default(opts, 'lambda_kkt', 1.0e-10);
opts = cmp_set_default(opts, 'nu', 1.0e-3);
opts = cmp_set_default(opts, 'velocityAmplitude', 0.20);
opts = cmp_set_default(opts, 'velocityGamma', 0.50);
opts = cmp_set_default(opts, 'pressureAmplitude', 0.05);
opts = cmp_set_default(opts, 'pressureGamma', 0.30);
opts = cmp_set_default(opts, 't0', 0.0);
opts = cmp_set_default(opts, 'dt', 0.02);
% T=2 long-time test by default: finalTime = dt*nSteps = 0.02*100 = 2.
opts = cmp_set_default(opts, 'nSteps', 100);
opts = cmp_set_default(opts, 'doCleanupProjection', true);
opts = cmp_set_default(opts, 'useBoundaryCompatibilityCorrection', true);
opts = cmp_set_default(opts, 'compatCorrectionStrength', 0.30);
opts = cmp_set_default(opts, 'compatCorrectionMethod', 'coeff-a1-boundary-jet');
opts = cmp_set_default(opts, 'compatCorrectionUseExactBoundary', true);
opts = cmp_set_default(opts, 'compatCorrectionReproject', true);
opts = cmp_set_default(opts, 'compatCorrectionBoundaryTol', 1.0e-8);
opts = cmp_set_default(opts, 'compatCorrectionAcceptOnlyIfImproved', true);
opts = cmp_set_default(opts, 'compatCorrectionMaxGlobalGrowth', 1.00);
opts = cmp_set_default(opts, 'compatCorrectionMaxDivGrowth', 2.00);
opts = cmp_set_default(opts, 'compatCorrectionDivAbsTol', 1.0e-7);
opts = cmp_set_default(opts, 'compatCorrectionCoeffReg', 1.0e-8);
opts = cmp_set_default(opts, 'compatCorrectionMaxCoeffRel', 5.0e-4);
opts = cmp_set_default(opts, 'compatCorrectionPasses', 2);
opts = cmp_set_default(opts, 'stageCleanupProjection', true);
opts = cmp_set_default(opts, 'stageVelocityCleanupProjection', false);
opts = cmp_set_default(opts, 'useBspfFilter', true);
opts = cmp_set_default(opts, 'filterType', 'exponential');   % 'exponential' or 'twoThirds'
opts = cmp_set_default(opts, 'filterAlpha', 12);
opts = cmp_set_default(opts, 'filterOrder', 16);
opts = cmp_set_default(opts, 'filterCutoff', 2/3);
opts = cmp_set_default(opts, 'filterFper', true);
opts = cmp_set_default(opts, 'filterA2A3', true);
opts = cmp_set_default(opts, 'filterA1', false);
opts = cmp_set_default(opts, 'filterRHS', true);
opts = cmp_set_default(opts, 'filterProjectedK', true);
opts = cmp_set_default(opts, 'filterVelocityState', false);
opts = cmp_set_default(opts, 'useProjectedKNormalBoundaryCorrection', true);
opts = cmp_set_default(opts, 'projectedKNormalStrength', 0.3);
opts = cmp_set_default(opts, 'projectedKNormalCoeffReg', 1.0e-8);
opts = cmp_set_default(opts, 'projectedKNormalMaxCoeffRel', 1.0e-4);
opts = cmp_set_default(opts, 'projectedKNormalReproject', true);
opts = cmp_set_default(opts, 'makeFigures', true);
% Print only every 10 steps by default; set opts.printEvery=Inf to print only first/final.
opts = cmp_set_default(opts, 'printEvery', 10);

Lx = opts.Lx; Ly = opts.Ly;
Nx = opts.Nx; Ny = opts.Ny;
x = linspace(0, Lx, Nx);
y = linspace(0, Ly, Ny);
[X, Y] = meshgrid(x, y);

fprintf('\n=== H2F-T2: full-BSPF projected-RHS RK4 long-time test with weak RHS/K-only filtering ===\n');
fprintf('Domain: Lx=%.6g, Ly=%.6g, grid=%dx%d, dt=%.6g, nSteps=%d, finalTime=%.6g\n', ...
    Lx, Ly, Nx, Ny, opts.dt, opts.nSteps, opts.t0 + opts.dt*opts.nSteps);
fprintf('nu=%.3e, final cleanupProjection=%d, stage-RHS cleanup=%d, stage-velocity cleanup=%d\n', opts.nu, opts.doCleanupProjection, opts.stageCleanupProjection, opts.stageVelocityCleanupProjection);
fprintf('Boundary compatibility correction: use=%d, method=%s, passes=%d, strength=%.3g, reproject=%d, bdTol=%.3e\n', ...
    opts.useBoundaryCompatibilityCorrection, opts.compatCorrectionMethod, opts.compatCorrectionPasses, opts.compatCorrectionStrength, opts.compatCorrectionReproject, opts.compatCorrectionBoundaryTol);
fprintf('BSPF/Fourier filter: use=%d, type=%s, alpha=%.3g, order=%g, cutoff=%.3g, filterFper=%d, filterA2A3=%d, filterA1=%d\n', ...
    opts.useBspfFilter, opts.filterType, opts.filterAlpha, opts.filterOrder, opts.filterCutoff, opts.filterFper, opts.filterA2A3, opts.filterA1);
fprintf('Filter placement: RHS=%d, projectedK=%d, velocityState=%d.  Default H2F does NOT filter stage/final velocity states.\n', ...
    opts.filterRHS, opts.filterProjectedK, opts.filterVelocityState);
fprintf('Projected-K normal-boundary correction: use=%d, strength=%.3g, maxCoeffRel=%.3e, reproject=%d\n', ...
    opts.useProjectedKNormalBoundaryCorrection, opts.projectedKNormalStrength, opts.projectedKNormalMaxCoeffRel, opts.projectedKNormalReproject);
fprintf('BSPF parameters: degB=%d, nbasis=%d, kmax=%d, r=%d, lambda=%.1e\n', ...
    opts.degB, opts.nbasis, opts.kmax, opts.r, opts.lambda_kkt);
fprintf('No-slip velocity envelope: %d\n', opts.useNoSlipVelocityEnvelope);
fprintf('Require BSPF-only derivative/evolution paths: %d\n', opts.requireBspfOnlyDerivatives);
fprintf('Boundary-constrained BSPF decomposition: %d\n', opts.useBoundaryConstrainedDecomposition);
fprintf('KKT endpoint boundary constraints in BSPF split: %d\n', opts.useKktEndpointBoundaryConstraints);
fprintf('Endpoint derivative estimator: method=%s, radius=%d, degree=%d\n', ...
    opts.endpointDerivativeMethod, opts.endpointDerivativeRadius, opts.endpointDerivativeDegree);
fprintf('Allow endpoint derivative stencil inside BSPF split: %d\n', opts.allowEndpointDerivativeStencilInBspfSplit);
fprintf('Progress printing: first step, final step, and every opts.printEvery=%g steps.\n\n', opts.printEvery);

h2_validate_bspf_only_derivative_options(opts);

% Build complete-BSPF representation cache and BSPF-KKT Poisson cache once.
tPre = tic;
repCache = bspf_scalar_representation_precompute(x, y, opts);
poissonParams = default_bspf_kkt_neumann_params(Nx, Ny);
poissonParams.verbose = false;
poissonParams.degB = opts.degB;
poissonParams.nbasis = opts.nbasis;
poissonParams.s_newbasis = opts.kmax;
poissonParams.optSplit.kmax = opts.kmax;
poissonParams.optSplit.r = opts.r;
poissonParams.optSplit.lambda_kkt = opts.lambda_kkt;
poissonCache = bspf_kkt_poisson_neumann_precompute(x, y, poissonParams);
precomputeTime = toc(tPre);
fprintf('Representation and Poisson caches built in %.3f s.\n\n', precomputeTime);

% Initial raw-exact velocity -> complete BSPF representation.
t = opts.t0;
exact0 = h1_velocity_exact(X, Y, t, Lx, Ly, opts.velocityAmplitude, opts.velocityGamma);
repU = h2_decompose_velocity_scalar(exact0.u, repCache, opts, 'u');
repV = h2_decompose_velocity_scalar(exact0.v, repCache, opts, 'v');
if opts.filterVelocityState
    [repU, repV] = h2_filter_velocity_pair(repU, repV, repCache, opts);
end

hist = h2_init_history(opts.nSteps);
[hist, initDiag] = h2_store_diagnostics(hist, 0, t, repU, repV, [], repCache, X, Y, x, y, opts);
initEvalU = eval_full_bspf_scalar_derivatives(repU, repCache);
initEvalV = eval_full_bspf_scalar_derivatives(repV, repCache);
initPointwiseDiagnostics = struct();
initPointwiseDiagnostics.uValue = h2_error_band_diagnostics(abs(initEvalU.F - exact0.u));
initPointwiseDiagnostics.vValue = h2_error_band_diagnostics(abs(initEvalV.F - exact0.v));
initPointwiseDiagnostics.uGrad = h2_error_band_diagnostics(hypot(initEvalU.Fx - exact0.ux, initEvalU.Fy - exact0.uy));
initPointwiseDiagnostics.vGrad = h2_error_band_diagnostics(hypot(initEvalV.Fx - exact0.vx, initEvalV.Fy - exact0.vy));
initPointwiseDiagnostics.uLap = h2_error_band_diagnostics(abs(initEvalU.Lap - exact0.lapu));
initPointwiseDiagnostics.vLap = h2_error_band_diagnostics(abs(initEvalV.Lap - exact0.lapv));

fprintf('Initial full-BSPF velocity rel L2 = %.6e, bd Linf = %.6e, div Linf = %.6e\n\n', ...
    initDiag.velRelL2, initDiag.bdVelLinf, initDiag.divLinf);

totalLoopTimer = tic;
lastStageInfo = [];
lastCleanupInfo = [];

for step = 1:opts.nSteps
    tStep = opts.t0 + (step-1)*opts.dt;
    dt = opts.dt;
    stepTimer = tic;

    % Projected-RHS RK4 stages in BSPF coefficient space.
    [k1u, k1v, info1] = h2_projected_rhs_bspf(repU, repV, tStep, repCache, poissonCache, X, Y, x, y, opts);

    s2u = h2_lincomb_reps({repU, k1u}, [1, 0.5*dt], 'u_stage2');
    s2v = h2_lincomb_reps({repV, k1v}, [1, 0.5*dt], 'v_stage2');
    if opts.filterVelocityState
        [s2u, s2v] = h2_filter_velocity_pair(s2u, s2v, repCache, opts);
    end
    if opts.stageVelocityCleanupProjection
        [s2u, s2v, stageVelClean2] = h2_cleanup_project_velocity(s2u, s2v, repCache, poissonCache, opts, 'velocity');
        if opts.filterVelocityState
            [s2u, s2v] = h2_filter_velocity_pair(s2u, s2v, repCache, opts);
        end
    else
        stageVelClean2 = [];
    end
    [k2u, k2v, info2] = h2_projected_rhs_bspf(s2u, s2v, tStep + 0.5*dt, repCache, poissonCache, X, Y, x, y, opts);

    s3u = h2_lincomb_reps({repU, k2u}, [1, 0.5*dt], 'u_stage3');
    s3v = h2_lincomb_reps({repV, k2v}, [1, 0.5*dt], 'v_stage3');
    if opts.filterVelocityState
        [s3u, s3v] = h2_filter_velocity_pair(s3u, s3v, repCache, opts);
    end
    if opts.stageVelocityCleanupProjection
        [s3u, s3v, stageVelClean3] = h2_cleanup_project_velocity(s3u, s3v, repCache, poissonCache, opts, 'velocity');
        if opts.filterVelocityState
            [s3u, s3v] = h2_filter_velocity_pair(s3u, s3v, repCache, opts);
        end
    else
        stageVelClean3 = [];
    end
    [k3u, k3v, info3] = h2_projected_rhs_bspf(s3u, s3v, tStep + 0.5*dt, repCache, poissonCache, X, Y, x, y, opts);

    s4u = h2_lincomb_reps({repU, k3u}, [1, dt], 'u_stage4');
    s4v = h2_lincomb_reps({repV, k3v}, [1, dt], 'v_stage4');
    if opts.filterVelocityState
        [s4u, s4v] = h2_filter_velocity_pair(s4u, s4v, repCache, opts);
    end
    if opts.stageVelocityCleanupProjection
        [s4u, s4v, stageVelClean4] = h2_cleanup_project_velocity(s4u, s4v, repCache, poissonCache, opts, 'velocity');
        if opts.filterVelocityState
            [s4u, s4v] = h2_filter_velocity_pair(s4u, s4v, repCache, opts);
        end
    else
        stageVelClean4 = [];
    end
    [k4u, k4v, info4] = h2_projected_rhs_bspf(s4u, s4v, tStep + dt, repCache, poissonCache, X, Y, x, y, opts);

    repU = h2_lincomb_reps({repU, k1u, k2u, k3u, k4u}, [1, dt/6, dt/3, dt/3, dt/6], 'u_next');
    repV = h2_lincomb_reps({repV, k1v, k2v, k3v, k4v}, [1, dt/6, dt/3, dt/3, dt/6], 'v_next');
    if opts.filterVelocityState
        [repU, repV] = h2_filter_velocity_pair(repU, repV, repCache, opts);
    end

    cleanupInfo = [];
    if opts.doCleanupProjection
        [repU, repV, cleanupInfo] = h2_cleanup_project_velocity(repU, repV, repCache, poissonCache, opts, 'velocity');
        if opts.filterVelocityState
            [repU, repV] = h2_filter_velocity_pair(repU, repV, repCache, opts);
        end
    end

    tNew = tStep + dt;
    compatInfo = [];
    if opts.useBoundaryCompatibilityCorrection
        compatInfo = struct('passes', {{}});
        for iCompatPass = 1:max(1, opts.compatCorrectionPasses)
            [repUtrial, repVtrial, passInfo] = h2_apply_boundary_compatibility_correction( ...
                repU, repV, repCache, poissonCache, X, Y, tNew, Lx, Ly, opts);
            passInfo.pass = iCompatPass;
            compatInfo.passes{end+1} = passInfo;
            repU = repUtrial;
            repV = repVtrial;
            if (~isfield(passInfo, 'wasAccepted') || ~passInfo.wasAccepted) && ...
               (~isfield(passInfo, 'low23WasAccepted') || ~passInfo.low23WasAccepted)
                break;
            end
        end
        if opts.filterVelocityState
            [repU, repV] = h2_filter_velocity_pair(repU, repV, repCache, opts);
        end
        if isempty(cleanupInfo)
            cleanupInfo = struct();
        end
        cleanupInfo.compatCorrection = compatInfo;
    end

    solveTime = toc(stepTimer);
    [hist, diag] = h2_store_diagnostics(hist, step, tNew, repU, repV, cleanupInfo, repCache, X, Y, x, y, opts);
    hist.solveTime(step+1) = solveTime;

    lastStageInfo = struct('k1',info1,'k2',info2,'k3',info3,'k4',info4, ...
        'stageVelClean2',stageVelClean2,'stageVelClean3',stageVelClean3,'stageVelClean4',stageVelClean4);
    lastCleanupInfo = cleanupInfo;

    doPrint = (step == 1) || (step == opts.nSteps) || ...
              (isfinite(opts.printEvery) && opts.printEvery > 0 && mod(step, opts.printEvery) == 0);
    if doPrint
        fprintf('step %3d/%3d t=%.6g | velRel=%.3e velInf=%.3e | bd=%.3e div=%.3e | cleanupGrad=%.3e | %.2fs\n', ...
            step, opts.nSteps, tNew, diag.velRelL2, diag.velLinf, diag.bdVelLinf, ...
            diag.divLinf, diag.cleanupGradL2, solveTime);
    end
end

totalLoopTime = toc(totalLoopTimer);

fprintf('\n======================== H2F-T2 full-BSPF RK4 weak RHS/K-filter long-time summary ========================\n');
fprintf('final time                    = %.6g\n', hist.time(end));
fprintf('final velocity rel L2         = %.6e\n', hist.velRelL2(end));
fprintf('final velocity Linf           = %.6e\n', hist.velLinf(end));
fprintf('max velocity rel L2           = %.6e\n', max(hist.velRelL2));
fprintf('final boundary velocity Linf  = %.6e\n', hist.bdVelLinf(end));
fprintf('max boundary velocity Linf    = %.6e\n', max(hist.bdVelLinf));
fprintf('final full boundary vel Linf  = %.6e\n', hist.fullBdVelLinf(end));
fprintf('final tangent boundary Linf   = %.6e\n', hist.tanBdVelLinf(end));
fprintf('final divergence Linf         = %.6e\n', hist.divLinf(end));
fprintf('max divergence Linf           = %.6e\n', max(hist.divLinf));
fprintf('final cleanup grad L2         = %.6e\n', hist.cleanupGradL2(end));
fprintf('precompute time               = %.3f s\n', precomputeTime);
fprintf('time loop wall time           = %.3f s\n', totalLoopTime);
fprintf('=========================================================================\n\n');

if opts.makeFigures
    evalU = eval_full_bspf_scalar_derivatives(repU, repCache);
    evalV = eval_full_bspf_scalar_derivatives(repV, repCache);
    exactF = h1_velocity_exact(X, Y, hist.time(end), Lx, Ly, opts.velocityAmplitude, opts.velocityGamma);
    divFinal = evalU.Fx + evalV.Fy;
    h2_make_figures(hist, x, y, evalU, evalV, exactF, divFinal);
end

results = struct();
results.opts = opts;
results.repU = repU;
results.repV = repV;
results.history = hist;
results.lastStageInfo = lastStageInfo;
results.lastCleanupInfo = lastCleanupInfo;
results.initialPointwiseDiagnostics = initPointwiseDiagnostics;
results.precomputeTime = precomputeTime;
results.totalLoopTime = totalLoopTime;
results.cacheInfo = struct('Nx',Nx,'Ny',Ny,'degB',opts.degB,'nbasis',opts.nbasis, ...
    'kmax',opts.kmax,'r',opts.r,'endpointDerivativeMethod',opts.endpointDerivativeMethod, ...
    'endpointDerivativeRadius',opts.endpointDerivativeRadius,'endpointDerivativeDegree',opts.endpointDerivativeDegree, ...
    'useNoSlipVelocityEnvelope',opts.useNoSlipVelocityEnvelope, ...
    'useBoundaryConstrainedDecomposition',opts.useBoundaryConstrainedDecomposition, ...
    'useKktEndpointBoundaryConstraints',opts.useKktEndpointBoundaryConstraints, ...
    'lambda_kkt',opts.lambda_kkt,'stageCleanupProjection',opts.stageCleanupProjection,'stageVelocityCleanupProjection',opts.stageVelocityCleanupProjection,'useBspfFilter',opts.useBspfFilter,'filterType',opts.filterType,'filterAlpha',opts.filterAlpha,'filterOrder',opts.filterOrder,'filterRHS',opts.filterRHS,'filterProjectedK',opts.filterProjectedK,'filterVelocityState',opts.filterVelocityState);
evalU_diag = eval_full_bspf_scalar_derivatives(repU, repCache);
evalV_diag = eval_full_bspf_scalar_derivatives(repV, repCache);
exact_diag = h1_velocity_exact(X, Y, hist.time(end), Lx, Ly, opts.velocityAmplitude, opts.velocityGamma);
errMag_diag = hypot(evalU_diag.F - exact_diag.u, evalV_diag.F - exact_diag.v);
results.finalPointwiseDiagnostics = h2_error_band_diagnostics(errMag_diag);
end


function h2_validate_bspf_only_derivative_options(opts)
if ~isfield(opts, 'requireBspfOnlyDerivatives') || ~opts.requireBspfOnlyDerivatives
    return;
end

if isfield(opts, 'endpointDerivativeMethod') && strcmpi(opts.endpointDerivativeMethod, 'fd')
    error('BSPF-only derivative mode blocks endpointDerivativeMethod=''fd''.');
end

if isfield(opts, 'allowEndpointDerivativeStencilInBspfSplit') && ~opts.allowEndpointDerivativeStencilInBspfSplit
    if isfield(opts, 'endpointDerivativeMethod') && any(strcmpi(opts.endpointDerivativeMethod, {'fd','local-poly-qr'}))
        error(['Strict no-stencil mode blocks endpointDerivativeMethod=''%s''. ', ...
            'The current BSPF split still estimates endpoint derivatives from grid data; ', ...
            'a pure BSPF-coefficient endpoint constraint split is required next.'], opts.endpointDerivativeMethod);
    end
end
end


function [repUcorr, repVcorr, info] = h2_apply_boundary_compatibility_correction(repU, repV, repCache, poissonCache, X, Y, t, Lx, Ly, opts)
if ~isfield(opts, 'compatCorrectionUseExactBoundary') || ~opts.compatCorrectionUseExactBoundary
    error('Current boundary compatibility correction prototype requires exact MMS boundary data.');
end

[repUcorr, repVcorr, info] = h2_apply_a1_boundary_jet_compatibility_correction( ...
    repU, repV, repCache, poissonCache, X, Y, t, Lx, Ly, opts);
end


function [repUcorr, repVcorr, info] = h2_apply_a1_boundary_jet_compatibility_correction(repU, repV, repCache, poissonCache, X, Y, t, Lx, Ly, opts)
evalU = eval_full_bspf_scalar_derivatives(repU, repCache);
evalV = eval_full_bspf_scalar_derivatives(repV, repCache);
exact = h1_velocity_exact(X, Y, t, Lx, Ly, opts.velocityAmplitude, opts.velocityGamma);

uErr = evalU.F - exact.u;
vErr = evalV.F - exact.v;
rawErr = hypot(uErr, vErr);
bdErr = max([rawErr(1,:), rawErr(end,:), rawErr(:,1).', rawErr(:,end).']);
rawDiv = max(abs(evalU.Fx + evalV.Fy), [], 'all');

info = struct();
info.method = 'coeff-a1-boundary-jet';
info.strength = opts.compatCorrectionStrength;
info.rawGlobalLinf = max(rawErr(:));
info.rawGlobalL2 = sqrt(mean(rawErr(:).^2));
info.boundaryLinf = bdErr;
info.rawDivLinf = rawDiv;
info.wasApplied = bdErr >= opts.compatCorrectionBoundaryTol;
info.wasAccepted = false;

repUcorr = repU;
repVcorr = repV;
if ~info.wasApplied
    info.corrGlobalLinf = info.rawGlobalLinf;
    info.corrGlobalL2 = info.rawGlobalL2;
    info.corrBoundaryLinf = bdErr;
    info.corrDivLinf = rawDiv;
    info.coeffChangeRel = 0;
    return;
end

pc = h2_a1_boundary_jet_precompute(repCache);
uBd = uErr(pc.boundaryMask);
vBd = vErr(pc.boundaryMask);

V = pc.Vbd;
reg = opts.compatCorrectionCoeffReg;
G = V.' * V + reg * speye(size(V,2));
du = -opts.compatCorrectionStrength * (G \ (V.' * uBd));
dv = -opts.compatCorrectionStrength * (G \ (V.' * vBd));

baseNorm = max(norm([repU.A1(:); repV.A1(:)]), 1);
stepNorm = norm([du; dv]);
scale = min(1, opts.compatCorrectionMaxCoeffRel * baseNorm / max(stepNorm, eps));
du = scale * du;
dv = scale * dv;

trialU = h2_apply_a1_boundary_jet_delta(repU, pc.dofs, du);
trialV = h2_apply_a1_boundary_jet_delta(repV, pc.dofs, dv);

if opts.compatCorrectionReproject
    [trialU, trialV, compatProjectionInfo] = h2_cleanup_project_velocity(trialU, trialV, repCache, poissonCache, opts, 'velocity');
    info.reprojectCleanup = compatProjectionInfo;
end

evalUcorr = eval_full_bspf_scalar_derivatives(trialU, repCache);
evalVcorr = eval_full_bspf_scalar_derivatives(trialV, repCache);
corrErr = hypot(evalUcorr.F - exact.u, evalVcorr.F - exact.v);
corrBdErr = max([corrErr(1,:), corrErr(end,:), corrErr(:,1).', corrErr(:,end).']);
corrDiv = max(abs(evalUcorr.Fx + evalVcorr.Fy), [], 'all');

info.corrGlobalLinf = max(corrErr(:));
info.corrGlobalL2 = sqrt(mean(corrErr(:).^2));
info.corrBoundaryLinf = corrBdErr;
info.corrDivLinf = corrDiv;
info.coeffChangeRel = norm([du; dv]) / baseNorm;
info.scaleLimiter = scale;
info.nDofs = numel(pc.dofs);

if opts.compatCorrectionAcceptOnlyIfImproved
    divLimit = max(opts.compatCorrectionDivAbsTol, opts.compatCorrectionMaxDivGrowth * max(rawDiv, eps));
    accept = (corrBdErr <= bdErr) && ...
             (info.corrGlobalLinf <= opts.compatCorrectionMaxGlobalGrowth * info.rawGlobalLinf) && ...
             (corrDiv <= divLimit);
    if ~accept
        info.wasApplied = false;
        info.wasAccepted = false;
        info.rejectReason = sprintf('bd %.3e->%.3e, global %.3e->%.3e, div %.3e->%.3e limit %.3e', ...
            bdErr, corrBdErr, info.rawGlobalLinf, info.corrGlobalLinf, rawDiv, corrDiv, divLimit);
        return;
    end
end

repUcorr = trialU;
repVcorr = trialV;
info.wasAccepted = true;
end


function pc = h2_a1_boundary_jet_precompute(cache)
[Ny, Nx] = deal(cache.Ny, cache.Nx);
boundaryMask = false(Ny, Nx);
boundaryMask(1,:) = true; boundaryMask(end,:) = true;
boundaryMask(:,1) = true; boundaryMask(:,end) = true;

tolX = 1.0e-12 * max(1, max(abs(cache.Bx(:))));
tolY = 1.0e-12 * max(1, max(abs(cache.By(:))));
edgeX = find(abs(cache.Bx(:,1)) > tolX | abs(cache.Bx(:,end)) > tolX);
edgeY = find(abs(cache.By(:,1)) > tolY | abs(cache.By(:,end)) > tolY);

dofs = struct('jx', {}, 'jy', {});
for jy = 1:cache.nbasis
    for jx = edgeX(:).'
        dofs(end+1) = struct('jx', jx, 'jy', jy); %#ok<AGROW>
    end
end
for jy = edgeY(:).'
    for jx = 1:cache.nbasis
        if ~ismember(jx, edgeX)
            dofs(end+1) = struct('jx', jx, 'jy', jy); %#ok<AGROW>
        end
    end
end

Vbd = zeros(nnz(boundaryMask), numel(dofs));
for q = 1:numel(dofs)
    phi = cache.By(dofs(q).jy,:).' * cache.Bx(dofs(q).jx,:);
    Vbd(:, q) = phi(boundaryMask);
end

pc = struct();
pc.boundaryMask = boundaryMask;
pc.Vbd = sparse(Vbd);
pc.dofs = dofs;
pc.edgeX = edgeX;
pc.edgeY = edgeY;
end


function rep = h2_apply_a1_boundary_jet_delta(rep, dofs, delta)
for q = 1:numel(dofs)
    if delta(q) ~= 0
        rep.A1(dofs(q).jx, dofs(q).jy) = rep.A1(dofs(q).jx, dofs(q).jy) + delta(q);
    end
end
end


function diag = h2_error_band_diagnostics(errMag)
[Ny, Nx] = size(errMag);
[J, I] = ndgrid(1:Ny, 1:Nx);
distToBoundary = min(cat(3, I - 1, Nx - I, J - 1, Ny - J), [], 3);
bands = [0, 1, 2, 3, 5, 10];
diag = struct();
diag.globalMax = max(errMag(:));
diag.globalMean = mean(errMag(:));
diag.globalRms = sqrt(mean(errMag(:).^2));
diag.bands = struct([]);
for k = 1:numel(bands)
    width = bands(k);
    boundaryMask = distToBoundary <= width;
    interiorMask = distToBoundary > width;
    diag.bands(k).boundaryLayers = width;
    diag.bands(k).boundaryMax = max(errMag(boundaryMask));
    diag.bands(k).boundaryMean = mean(errMag(boundaryMask));
    diag.bands(k).boundaryRms = sqrt(mean(errMag(boundaryMask).^2));
    diag.bands(k).interiorMax = max(errMag(interiorMask));
    diag.bands(k).interiorMean = mean(errMag(interiorMask));
    diag.bands(k).interiorRms = sqrt(mean(errMag(interiorMask).^2));
    diag.bands(k).maxRatio = diag.bands(k).boundaryMax / diag.bands(k).interiorMax;
    diag.bands(k).rmsRatio = diag.bands(k).boundaryRms / diag.bands(k).interiorRms;
end
end


function [repKu, repKv, info] = h2_projected_rhs_bspf(repU, repV, t, repCache, poissonCache, X, Y, x, y, opts)
% Build and project the instantaneous RHS using complete BSPF derivatives.
% H2C adds a second projection after K is re-represented in BSPF form:
%   R -> P(R) -> BSPF representation -> P(BSPF representation)
% This is intended to remove divergence/normal-velocity errors reintroduced
% by finite-dimensional BSPF re-representation of the projected RHS.

evalU = eval_full_bspf_scalar_derivatives(repU, repCache);
evalV = eval_full_bspf_scalar_derivatives(repV, repCache);

Lx = opts.Lx; Ly = opts.Ly;
exact = h1_velocity_exact(X, Y, t, Lx, Ly, opts.velocityAmplitude, opts.velocityGamma);
pres  = h1_pressure_exact(X, Y, t, Lx, Ly, opts.pressureAmplitude, opts.pressureGamma);

convU_exact = exact.u .* exact.ux + exact.v .* exact.uy;
convV_exact = exact.u .* exact.vx + exact.v .* exact.vy;
forceU = exact.ut + convU_exact - opts.nu * exact.lapu + pres.px;
forceV = exact.vt + convV_exact - opts.nu * exact.lapv + pres.py;

convU = evalU.F .* evalU.Fx + evalV.F .* evalU.Fy;
convV = evalU.F .* evalV.Fx + evalV.F .* evalV.Fy;
Ru_grid = -convU + opts.nu * evalU.Lap + forceU;
Rv_grid = -convV + opts.nu * evalV.Lap + forceV;

% Represent RHS in complete BSPF form.
repRu = decompose_scalar_to_full_bspf(Ru_grid, repCache, 'Ru');
repRv = decompose_scalar_to_full_bspf(Rv_grid, repCache, 'Rv');
if opts.filterRHS
    [repRu, repRv] = h2_filter_velocity_pair(repRu, repRv, repCache, opts);
end
evalRu = eval_full_bspf_scalar_derivatives(repRu, repCache);
evalRv = eval_full_bspf_scalar_derivatives(repRv, repCache);

% First projected-RHS projection.
rhsPi = evalRu.Fx + evalRv.Fy;
qL = -evalRu.F(:,1);
qR =  evalRu.F(:,end);
qB = -evalRv.F(1,:);
qT =  evalRv.F(end,:);

[Pi, infoPi] = bspf_kkt_poisson_neumann_apply_cached(rhsPi, qL, qR, qB, qT, poissonCache, 0);
repPi = decompose_scalar_to_full_bspf(Pi, repCache, 'pi');
evalPi = eval_full_bspf_scalar_derivatives(repPi, repCache);

Ku_grid = evalRu.F - evalPi.Fx;
Kv_grid = evalRv.F - evalPi.Fy;

% Re-represent projected RHS in BSPF form. This operation may reintroduce a
% small divergence and boundary-normal component.
repKu = h2_decompose_velocity_scalar(Ku_grid, repCache, opts, 'Ku_rawrep', false);
repKv = h2_decompose_velocity_scalar(Kv_grid, repCache, opts, 'Kv_rawrep', false);
if opts.filterProjectedK
    [repKu, repKv] = h2_filter_velocity_pair(repKu, repKv, repCache, opts);
end

stageCleanupInfo = [];
if isfield(opts, 'stageCleanupProjection') && opts.stageCleanupProjection
    [repKu, repKv, stageCleanupInfo] = h2_cleanup_project_velocity(repKu, repKv, repCache, poissonCache, opts, 'projectedRhs');
    repKu.name = 'Ku_stageclean';
    repKv.name = 'Kv_stageclean';
    if opts.filterProjectedK
        [repKu, repKv] = h2_filter_velocity_pair(repKu, repKv, repCache, opts);
    end
end

kNormalInfo = [];
if isfield(opts, 'useProjectedKNormalBoundaryCorrection') && opts.useProjectedKNormalBoundaryCorrection
    [repKu, repKv, kNormalInfo] = h2_apply_projected_k_normal_boundary_correction( ...
        repKu, repKv, repCache, poissonCache, opts);
    if opts.filterProjectedK
        [repKu, repKv] = h2_filter_velocity_pair(repKu, repKv, repCache, opts);
    end
end

if nargout > 2
    evalKu = eval_full_bspf_scalar_derivatives(repKu, repCache);
    evalKv = eval_full_bspf_scalar_derivatives(repKv, repCache);
    Kdiv = evalKu.Fx + evalKv.Fy;
    target = h1_vector_metrics(evalKu.F, evalKv.F, exact.ut, exact.vt, x, y);
    bd = h1_boundary_vector_norm(evalKu.F, evalKv.F);

    rawKdiv = rhsPi - evalPi.Lap;
    rawTarget = h1_vector_metrics(Ku_grid, Kv_grid, exact.ut, exact.vt, x, y);
    rawBd = h1_boundary_vector_norm(Ku_grid, Kv_grid);
    rawRErrBand = h2_error_band_diagnostics(hypot(Ru_grid - (exact.ut + pres.px), Rv_grid - (exact.vt + pres.py)));
    repRErrBand = h2_error_band_diagnostics(hypot(evalRu.F - (exact.ut + pres.px), evalRv.F - (exact.vt + pres.py)));
    pressureGradErrBand = h2_error_band_diagnostics(hypot(evalPi.Fx - pres.px, evalPi.Fy - pres.py));
    pressureRerepBand = h2_error_band_diagnostics(abs(evalPi.F - Pi));
    rawKErrBand = h2_error_band_diagnostics(hypot(Ku_grid - exact.ut, Kv_grid - exact.vt));
    repKErrBand = h2_error_band_diagnostics(hypot(evalKu.F - exact.ut, evalKv.F - exact.vt));
    kRerepBand = h2_error_band_diagnostics(hypot(evalKu.F - Ku_grid, evalKv.F - Kv_grid));

    info = struct();
    info.Pi = Pi;
    info.repPi = repPi;
    info.infoPi = infoPi;
    info.stageCleanupInfo = stageCleanupInfo;
    info.kNormalBoundaryInfo = kNormalInfo;
    info.KRelL2_vs_ut = target.relL2;
    info.KLinf_vs_ut = target.linf;
    info.KBoundaryLinf = bd.linf;
    info.KDivLinf = max(abs(Kdiv(:)));
    info.rawKRelL2_vs_ut = rawTarget.relL2;
    info.rawKLinf_vs_ut = rawTarget.linf;
    info.rawKBoundaryLinf = rawBd.linf;
    info.rawKDivLinf = max(abs(rawKdiv(:)));
    info.rawRErrBand = rawRErrBand;
    info.repRErrBand = repRErrBand;
    info.pressureGradErrBand = pressureGradErrBand;
    info.pressureRerepBand = pressureRerepBand;
    info.rawKErrBand = rawKErrBand;
    info.repKErrBand = repKErrBand;
    info.kRerepBand = kRerepBand;
    info.RuSplitLinf = repRu.dbg.reconstruction_linf;
    info.RvSplitLinf = repRv.dbg.reconstruction_linf;
end
end


function [repUclean, repVclean, info] = h2_cleanup_project_velocity(repU, repV, repCache, poissonCache, opts, context)
if nargin < 5 || isempty(opts)
    opts = struct();
end
if nargin < 6 || isempty(context)
    context = 'velocity';
end

evalU = eval_full_bspf_scalar_derivatives(repU, repCache);
evalV = eval_full_bspf_scalar_derivatives(repV, repCache);
fullBoundaryBeforeNoSlip = h2_full_boundary_vector_norm(evalU.F, evalV.F);

rhsPhi = evalU.Fx + evalV.Fy;
qL = -evalU.F(:,1);
qR =  evalU.F(:,end);
qB = -evalV.F(1,:);
qT =  evalV.F(end,:);

[Phi, infoPhi] = bspf_kkt_poisson_neumann_apply_cached(rhsPhi, qL, qR, qB, qT, poissonCache, 0);
repPhi = decompose_scalar_to_full_bspf(Phi, repCache, 'cleanupPhi');
evalPhi = eval_full_bspf_scalar_derivatives(repPhi, repCache);

uClean = evalU.F - evalPhi.Fx;
vClean = evalV.F - evalPhi.Fy;
fullBoundaryBeforePostNoSlip = h2_full_boundary_vector_norm(uClean, vClean);
fullBoundaryAfterNoSlip = fullBoundaryBeforePostNoSlip;

constrainCleanVelocity = strcmpi(context, 'velocity');
repUclean = h2_decompose_velocity_scalar(uClean, repCache, opts, 'u_clean', constrainCleanVelocity);
repVclean = h2_decompose_velocity_scalar(vClean, repCache, opts, 'v_clean', constrainCleanVelocity);
evalUclean = eval_full_bspf_scalar_derivatives(repUclean, repCache);
evalVclean = eval_full_bspf_scalar_derivatives(repVclean, repCache);
fullBoundaryAfterRepr = h2_full_boundary_vector_norm(evalUclean.F, evalVclean.F);

divAfter = rhsPhi - evalPhi.Lap;
info = struct();
info.Phi = Phi;
info.repPhi = repPhi;
info.infoPhi = infoPhi;
info.gradL2 = sqrt(weighted_mean_2d(evalPhi.Fx.^2 + evalPhi.Fy.^2, repCache.x, repCache.y));
info.phiL2 = sqrt(weighted_mean_2d(evalPhi.F.^2, repCache.x, repCache.y));
info.divLinfBefore = max(abs(rhsPhi(:)));
info.divLinfAfter = max(abs(divAfter(:)));
info.neumannLinf = max([abs((-evalPhi.Fx(:,1)-qL(:))); abs((evalPhi.Fx(:,end)-qR(:))); ...
    abs((-evalPhi.Fy(1,:).'-qB(:))); abs((evalPhi.Fy(end,:).'-qT(:)))]);
info.cleanupRerepBand = h2_error_band_diagnostics(hypot(evalUclean.F - uClean, evalVclean.F - vClean));
info.fullBoundaryBeforeNoSlip = fullBoundaryBeforeNoSlip;
info.fullBoundaryBeforePostNoSlip = fullBoundaryBeforePostNoSlip;
info.fullBoundaryAfterNoSlip = fullBoundaryAfterNoSlip;
info.fullBoundaryAfterRepr = fullBoundaryAfterRepr;
info.noSlipBoundaryApplied = false;
info.cleanupContext = context;
end


function [repU, repV] = h2_filter_velocity_pair(repU, repV, repCache, opts)
repU = h2_filter_bspf_rep(repU, repCache, opts);
repV = h2_filter_bspf_rep(repV, repCache, opts);
end


function rep = h2_filter_bspf_rep(rep, repCache, opts)
% BSPF/Fourier filter for complete BSPF representation.
% It damps high Fourier modes in:
%   - rep.f_per: 2D periodic residual
%   - rep.A2: B_y times Fourier-x coefficients
%   - rep.A3: B_x times Fourier-y coefficients
% Optionally, rep.A1 B-spline coefficients can be softly filtered in basis-index space.

if ~isfield(opts, 'useBspfFilter') || ~opts.useBspfFilter
    return;
end

if isfield(opts,'filterFper') && opts.filterFper && isfield(rep,'f_per') && ~isempty(rep.f_per)
    f0 = rep.f_per(1:repCache.Ny0, 1:repCache.Nx0);
    Fhat = fft2(f0);
    sigma2 = h2_filter_sigma_2d(repCache.Nx0, repCache.Ny0, opts);
    f0f = real(ifft2(Fhat .* sigma2));
    rep.f_per = embed_periodic_full(f0f);
end

if isfield(opts,'filterA2A3') && opts.filterA2A3
    if isfield(rep,'A2') && ~isempty(rep.A2)
        sx = h2_filter_sigma_1d(repCache.Nx0, opts);
        rep.A2 = rep.A2 .* sx(:).';
    end
    if isfield(rep,'A3') && ~isempty(rep.A3)
        sy = h2_filter_sigma_1d(repCache.Ny0, opts);
        rep.A3 = rep.A3 .* sy(:).';
    end
end

if isfield(opts,'filterA1') && opts.filterA1 && isfield(rep,'A1') && ~isempty(rep.A1)
    nbx = size(rep.A1,1);
    nby = size(rep.A1,2);
    sx = h2_filter_sigma_bspline(nbx, opts);
    sy = h2_filter_sigma_bspline(nby, opts);
    rep.A1 = rep.A1 .* (sx(:) * sy(:).');
end
end


function sigma = h2_filter_sigma_1d(N, opts)
idx = [0:floor(N/2), -ceil(N/2)+1:-1];
rho = abs(idx) / max(1, max(abs(idx)));
sigma = h2_filter_sigma_from_rho(rho, opts);
end


function sigma2 = h2_filter_sigma_2d(Nx, Ny, opts)
kx = [0:floor(Nx/2), -ceil(Nx/2)+1:-1];
ky = [0:floor(Ny/2), -ceil(Ny/2)+1:-1];
[KX, KY] = meshgrid(kx, ky);
rho = sqrt((KX/max(1,max(abs(kx)))).^2 + (KY/max(1,max(abs(ky)))).^2);
rho = min(1, rho);
sigma2 = h2_filter_sigma_from_rho(rho, opts);
end


function sigma = h2_filter_sigma_bspline(n, opts)
if n <= 1
    sigma = 1;
else
    rho = (0:n-1) / (n-1);
    sigma = h2_filter_sigma_from_rho(rho, opts);
end
end


function sigma = h2_filter_sigma_from_rho(rho, opts)
filterType = 'exponential';
if isfield(opts,'filterType') && ~isempty(opts.filterType)
    filterType = opts.filterType;
end

switch lower(filterType)
    case {'twothirds','2/3','two_thirds'}
        cutoff = 2/3;
        if isfield(opts,'filterCutoff') && ~isempty(opts.filterCutoff)
            cutoff = opts.filterCutoff;
        end
        sigma = double(rho <= cutoff);
    case {'exponential','exp'}
        alpha = 36;
        p = 8;
        if isfield(opts,'filterAlpha') && ~isempty(opts.filterAlpha), alpha = opts.filterAlpha; end
        if isfield(opts,'filterOrder') && ~isempty(opts.filterOrder), p = opts.filterOrder; end
        sigma = exp(-alpha * rho.^p);
    otherwise
        error('Unknown filterType: %s', filterType);
end
% Preserve exact mean/DC mode.
sigma(rho == 0) = 1;
end


function rep = h2_lincomb_reps(repList, weights, name)
if numel(repList) ~= numel(weights)
    error('h2_lincomb_reps: repList and weights must have same length.');
end
rep = repList{1};
rep.name = name;
rep.A1 = weights(1) * repList{1}.A1;
rep.A2 = weights(1) * repList{1}.A2;
rep.A3 = weights(1) * repList{1}.A3;
rep.f_per = weights(1) * repList{1}.f_per;
for k = 2:numel(repList)
    rep.A1 = rep.A1 + weights(k) * repList{k}.A1;
    rep.A2 = rep.A2 + weights(k) * repList{k}.A2;
    rep.A3 = rep.A3 + weights(k) * repList{k}.A3;
    rep.f_per = rep.f_per + weights(k) * repList{k}.f_per;
end
rep.dbg = struct('linear_combination', true);
end


function hist = h2_init_history(nSteps)
hist = struct();
hist.step = (0:nSteps).';
hist.time = NaN(nSteps+1,1);
hist.velRelL2 = NaN(nSteps+1,1);
hist.velLinf = NaN(nSteps+1,1);
hist.uRelL2 = NaN(nSteps+1,1);
hist.vRelL2 = NaN(nSteps+1,1);
hist.bdVelLinf = NaN(nSteps+1,1);
hist.bdVelL2 = NaN(nSteps+1,1);
hist.fullBdVelLinf = NaN(nSteps+1,1);
hist.fullBdVelL2 = NaN(nSteps+1,1);
hist.tanBdVelLinf = NaN(nSteps+1,1);
hist.tanBdVelL2 = NaN(nSteps+1,1);
hist.divLinf = NaN(nSteps+1,1);
hist.divL2 = NaN(nSteps+1,1);
hist.cleanupGradL2 = NaN(nSteps+1,1);
hist.cleanupPhiL2 = NaN(nSteps+1,1);
hist.cleanupDivAfter = NaN(nSteps+1,1);
hist.solveTime = NaN(nSteps+1,1);
end


function [hist, diag] = h2_store_diagnostics(hist, step, t, repU, repV, cleanupInfo, repCache, X, Y, x, y, opts)
evalU = eval_full_bspf_scalar_derivatives(repU, repCache);
evalV = eval_full_bspf_scalar_derivatives(repV, repCache);
exact = h1_velocity_exact(X, Y, t, opts.Lx, opts.Ly, opts.velocityAmplitude, opts.velocityGamma);
vm = h1_vector_metrics(evalU.F, evalV.F, exact.u, exact.v, x, y);
uM = h1_rel_l2(evalU.F-exact.u, exact.u, x, y);
vM = h1_rel_l2(evalV.F-exact.v, exact.v, x, y);
bd = h1_boundary_vector_norm(evalU.F, evalV.F);
fullBd = h2_full_boundary_vector_norm(evalU.F, evalV.F);
divVal = evalU.Fx + evalV.Fy;
divLinf = max(abs(divVal(:)));
divL2 = sqrt(weighted_mean_2d(divVal.^2, x, y));

cleanupGrad = 0;
cleanupPhi = 0;
cleanupDivAfter = divLinf;
if ~isempty(cleanupInfo)
    cleanupGrad = cleanupInfo.gradL2;
    cleanupPhi = cleanupInfo.phiL2;
    cleanupDivAfter = cleanupInfo.divLinfAfter;
end

idx = step + 1;
hist.time(idx) = t;
hist.velRelL2(idx) = vm.relL2;
hist.velLinf(idx) = vm.linf;
hist.uRelL2(idx) = uM;
hist.vRelL2(idx) = vM;
hist.bdVelLinf(idx) = bd.linf;
hist.bdVelL2(idx) = bd.l2;
hist.fullBdVelLinf(idx) = fullBd.fullLinf;
hist.fullBdVelL2(idx) = fullBd.fullL2;
hist.tanBdVelLinf(idx) = fullBd.tangentLinf;
hist.tanBdVelL2(idx) = fullBd.tangentL2;
hist.divLinf(idx) = divLinf;
hist.divL2(idx) = divL2;
hist.cleanupGradL2(idx) = cleanupGrad;
hist.cleanupPhiL2(idx) = cleanupPhi;
hist.cleanupDivAfter(idx) = cleanupDivAfter;

diag = struct('velRelL2',vm.relL2,'velLinf',vm.linf,'bdVelLinf',bd.linf, ...
    'bdVelL2',bd.l2,'fullBdVelLinf',fullBd.fullLinf,'fullBdVelL2',fullBd.fullL2, ...
    'tanBdVelLinf',fullBd.tangentLinf,'tanBdVelL2',fullBd.tangentL2, ...
    'divLinf',divLinf,'divL2',divL2, ...
    'cleanupGradL2',cleanupGrad,'cleanupPhiL2',cleanupPhi);
end


function h2_make_figures(hist, x, y, evalU, evalV, exact, divFinal)
fontSize = 14;
figure('Color','w','Name','H2F-T2 full-BSPF projected-RHS RK4 history', ...
    'Units','normalized','Position',[0.04 0.06 0.88 0.80]);
tiledlayout(2,2,'TileSpacing','compact','Padding','compact');

nexttile;
semilogy(hist.time, max(hist.velRelL2, eps), 'o-', 'LineWidth',1.5); grid on
set(gca,'FontSize',fontSize); xlabel('time'); ylabel('relative L2')
title('raw-exact velocity error')

nexttile;
semilogy(hist.time, max(hist.bdVelLinf, eps), 'o-', 'LineWidth',1.5); grid on
set(gca,'FontSize',fontSize); xlabel('time'); ylabel('Linf')
title('boundary normal velocity')

nexttile;
semilogy(hist.time, max(hist.divLinf, eps), 'o-', 'LineWidth',1.5); grid on
set(gca,'FontSize',fontSize); xlabel('time'); ylabel('Linf')
title('BSPF divergence after cleanup')

nexttile;
semilogy(hist.time, max(hist.cleanupGradL2, eps), 'o-', 'LineWidth',1.5); grid on
set(gca,'FontSize',fontSize); xlabel('time'); ylabel('L2')
title('cleanup projection |grad phi|')

sgtitle('H2F-T2: full-BSPF projected-RHS RK4 long-time test with weak RHS/K-only filtering', ...
    'FontSize',fontSize+4,'FontWeight','bold')

figure('Color','w','Name','H2F-T2 final fields', ...
    'Units','normalized','Position',[0.05 0.08 0.88 0.76]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

nexttile;
imagesc(x,y,evalU.F); axis xy image; colorbar; set(gca,'FontSize',fontSize)
title('final BSPF u'); xlabel('x'); ylabel('y')

nexttile;
imagesc(x,y,evalV.F); axis xy image; colorbar; set(gca,'FontSize',fontSize)
title('final BSPF v'); xlabel('x'); ylabel('y')

nexttile;
imagesc(x,y,sqrt((evalU.F-exact.u).^2 + (evalV.F-exact.v).^2)); axis xy image; colorbar; set(gca,'FontSize',fontSize)
title('velocity error'); xlabel('x'); ylabel('y')

nexttile;
imagesc(x,y,evalU.Fx+evalV.Fy); axis xy image; colorbar; set(gca,'FontSize',fontSize)
title('divergence from BSPF coeffs'); xlabel('x'); ylabel('y')

nexttile;
imagesc(x,y,divFinal); axis xy image; colorbar; set(gca,'FontSize',fontSize)
title('final div field'); xlabel('x'); ylabel('y')

nexttile;
imagesc(x,y,abs(evalU.F-exact.u)+abs(evalV.F-exact.v)); axis xy image; colorbar; set(gca,'FontSize',fontSize)
title('|u error|+|v error|'); xlabel('x'); ylabel('y')

sgtitle('H2F-T2 final raw-exact diagnostics', 'FontSize',fontSize+4,'FontWeight','bold')
end


function exact = h1_velocity_exact(X, Y, t, Lx, Ly, A, gamma)
ax = pi/Lx;
ay = pi/Ly;
fac = A * exp(-gamma*t);

sx = sin(ax*X); sy = sin(ay*Y);
sin2x = sin(2*ax*X); cos2x = cos(2*ax*X);
sin2y = sin(2*ay*Y); cos2y = cos(2*ay*Y);

u = fac * ay * sx.^2 .* sin2y;
v = -fac * ax * sin2x .* sy.^2;

ux = fac * ax*ay * sin2x .* sin2y;
uy = 2*fac * ay^2 * sx.^2 .* cos2y;
vx = -2*fac * ax^2 * cos2x .* sy.^2;
vy = -fac * ax*ay * sin2x .* sin2y;

uxx = 2*fac * ax^2*ay * cos2x .* sin2y;
uyy = -4*fac * ay^3 * sx.^2 .* sin2y;
vxx = 4*fac * ax^3 * sin2x .* sy.^2;
vyy = -2*fac * ax*ay^2 * sin2x .* cos2y;

exact = struct();
exact.u = u; exact.v = v;
exact.ux = ux; exact.uy = uy; exact.vx = vx; exact.vy = vy;
exact.lapu = uxx + uyy;
exact.lapv = vxx + vyy;
exact.div = ux + vy;
exact.ut = -gamma * u;
exact.vt = -gamma * v;
end


function pres = h1_pressure_exact(X, Y, t, Lx, Ly, A, gamma)
a = 2*pi/Lx;
b = pi/Ly;
c = pi/Lx;
d = 2*pi/Ly;
fac = A * exp(-gamma*t);
base = sin(a*X).*cos(b*Y) + 0.25*cos(c*X).*sin(d*Y);
p = fac * base;
px = fac * (a*cos(a*X).*cos(b*Y) - 0.25*c*sin(c*X).*sin(d*Y));
py = fac * (-b*sin(a*X).*sin(b*Y) + 0.25*d*cos(c*X).*cos(d*Y));
lap = fac * (-(a^2+b^2)*sin(a*X).*cos(b*Y) - 0.25*(c^2+d^2)*cos(c*X).*sin(d*Y));
pres = struct('p',p,'px',px,'py',py,'lap',lap);
end


function m = h1_vector_metrics(Au, Av, Bu, Bv, x, y)
err2 = (Au-Bu).^2 + (Av-Bv).^2;
ref2 = Bu.^2 + Bv.^2;
m = struct();
m.relL2 = sqrt(weighted_mean_2d(err2, x, y)) / max(sqrt(weighted_mean_2d(ref2, x, y)), eps);
m.linf = max(sqrt(err2(:)));
end


function val = h1_rel_l2(err, ref, x, y)
val = sqrt(weighted_mean_2d(err.^2, x, y)) / max(sqrt(weighted_mean_2d(ref.^2, x, y)), eps);
end


function bd = h1_boundary_vector_norm(u, v)
left = abs(-u(:,1));
right = abs(u(:,end));
bottom = abs(-v(1,:)).';
top = abs(v(end,:)).';
allv = [left; right; bottom; top];
bd = struct('linf',max(allv),'l2',sqrt(mean(allv.^2)));
end


function bd = h2_full_boundary_vector_norm(u, v)
normalVals = [abs(u(:,1)); abs(u(:,end)); abs(v(1,:)).'; abs(v(end,:)).'];
tangentVals = [abs(v(:,1)); abs(v(:,end)); abs(u(1,:)).'; abs(u(end,:)).'];
fullVals = [hypot(u(:,1), v(:,1)); hypot(u(:,end), v(:,end)); ...
    hypot(u(1,:).', v(1,:).'); hypot(u(end,:).', v(end,:).')];
bd = struct();
bd.normalLinf = max(normalVals);
bd.normalL2 = sqrt(mean(normalVals.^2));
bd.tangentLinf = max(tangentVals);
bd.tangentL2 = sqrt(mean(tangentVals.^2));
bd.fullLinf = max(fullVals);
bd.fullL2 = sqrt(mean(fullVals.^2));
end


function cache = bspf_scalar_representation_precompute(x, y, opts)
x = x(:).'; y = y(:).';
Nx = numel(x); Ny = numel(y);
Lx = x(end)-x(1); Ly = y(end)-y(1);
Nx0 = Nx-1; Ny0 = Ny-1;

cache = struct();
cache.x = x; cache.y = y;
cache.Nx = Nx; cache.Ny = Ny; cache.Nx0 = Nx0; cache.Ny0 = Ny0;
cache.Lx = Lx; cache.Ly = Ly;
cache.degB = opts.degB;
cache.nbasis = opts.nbasis;
cache.optSplit = struct('kmax',opts.kmax,'r',opts.r,'lambda_kkt',opts.lambda_kkt, ...
    'degB',opts.degB,'showWaitbar',false,'endpointDerivativeMethod',opts.endpointDerivativeMethod, ...
    'endpointDerivativeRadius',opts.endpointDerivativeRadius,'endpointDerivativeDegree',opts.endpointDerivativeDegree);

[BxSpline, Bvals_x] = make_bspline_basis_values(x, opts.degB, opts.nbasis);
[BySpline, Bvals_y] = make_bspline_basis_values(y, opts.degB, opts.nbasis);
cache.BxSpline = BxSpline;
cache.BySpline = BySpline;
cache.Bvals_x = Bvals_x;
cache.Bvals_y = Bvals_y;
[cache.Bx, cache.dBx, cache.ddBx, cache.dddBx] = bspline_values_derivatives_physical(x, Lx, BxSpline);
[cache.By, cache.dBy, cache.ddBy, cache.dddBy] = bspline_values_derivatives_physical(y, Ly, BySpline);
[Xg, Yg] = meshgrid(x, y);
xb = (Xg - x(1)) .* (x(end) - Xg);
yb = (Yg - y(1)) .* (y(end) - Yg);
cache.noSlipG = xb .* yb;
cache.noSlipGx = (x(end) + x(1) - 2*Xg) .* yb;
cache.noSlipGy = xb .* (y(end) + y(1) - 2*Yg);
cache.noSlipLapG = -2*yb - 2*xb;
cache.splitX = bspf_kkt_1d_decompose_precompute(x, Bvals_x, opts.kmax, opts.r, opts.lambda_kkt, BxSpline, cache.optSplit);
cache.splitY = bspf_kkt_1d_decompose_precompute(y, Bvals_y, opts.kmax, opts.r, opts.lambda_kkt, BySpline, cache.optSplit);

cache.fft = spectral_poisson_2d_uniform_precompute(Nx0, Ny0, Lx, Ly);
end


function [B0, B1, B2, B3] = bspline_values_derivatives_physical(t, L, Bspline)
t = t(:).';
tau = (t - t(1)) / L;
B0 = fnval(Bspline, tau);
B1 = fnval(fnder(Bspline, 1), tau) / L;
B2 = fnval(fnder(Bspline, 2), tau) / (L^2);
B3 = fnval(fnder(Bspline, 3), tau) / (L^3);
end


function rep = decompose_scalar_to_full_bspf(F, cache, name, bc)
if nargin < 4
    bc = [];
end
constraintInfo = [];
if ~isempty(bc)
    [F, constraintInfo] = h2_apply_scalar_boundary_constraints(F, cache, bc);
end
[f_per, A1, A2, A3, dbg] = split2D_kkt_directional_N0_cached(F, cache, bc);
rep = struct();
rep.name = name;
rep.A1 = A1;
rep.A2 = A2;
rep.A3 = A3;
rep.f_per = f_per;
rep.dbg = dbg;
rep.boundaryConstraint = constraintInfo;
end


function rep = h2_decompose_velocity_scalar(F, cache, opts, name, constrainBoundary)
if nargin < 5 || isempty(constrainBoundary)
    constrainBoundary = true;
end
if isfield(opts, 'useNoSlipVelocityEnvelope') && opts.useNoSlipVelocityEnvelope
    W = h2_divide_by_noslip_envelope(F, cache);
    rep = decompose_scalar_to_full_bspf(W, cache, [name '_hat']);
    rep.name = name;
    rep.envelope = 'noslip-bubble';
else
    bc = [];
    if constrainBoundary && isfield(opts, 'useBoundaryConstrainedDecomposition') && opts.useBoundaryConstrainedDecomposition
        bc = h2_velocity_boundary_bc(opts);
    end
    rep = decompose_scalar_to_full_bspf(F, cache, name, bc);
end
end


function [repKuc, repKvc, info] = h2_apply_projected_k_normal_boundary_correction(repKu, repKv, repCache, poissonCache, opts)
evalKu = eval_full_bspf_scalar_derivatives(repKu, repCache);
evalKv = eval_full_bspf_scalar_derivatives(repKv, repCache);

rawNormal = h1_boundary_vector_norm(evalKu.F, evalKv.F);
rawDiv = max(abs(evalKu.Fx + evalKv.Fy), [], 'all');

info = struct();
info.method = 'projected-k-normal-boundary-a1';
info.rawNormalBoundaryLinf = rawNormal.linf;
info.rawDivLinf = rawDiv;
info.wasAccepted = false;

pc = h2_a1_normal_boundary_precompute(repCache);
uVals = [evalKu.F(:,1); evalKu.F(:,end)];
vVals = [evalKv.F(1,:).'; evalKv.F(end,:).'];

Gu = pc.Vu.' * pc.Vu + opts.projectedKNormalCoeffReg * speye(size(pc.Vu,2));
Gv = pc.Vv.' * pc.Vv + opts.projectedKNormalCoeffReg * speye(size(pc.Vv,2));
du = -opts.projectedKNormalStrength * (Gu \ (pc.Vu.' * uVals));
dv = -opts.projectedKNormalStrength * (Gv \ (pc.Vv.' * vVals));

baseNorm = max(norm([repKu.A1(:); repKv.A1(:)]), 1);
stepNorm = norm([du; dv]);
scale = min(1, opts.projectedKNormalMaxCoeffRel * baseNorm / max(stepNorm, eps));
du = scale * du;
dv = scale * dv;

trialU = h2_apply_a1_boundary_jet_delta(repKu, pc.uDofs, du);
trialV = h2_apply_a1_boundary_jet_delta(repKv, pc.vDofs, dv);

if opts.projectedKNormalReproject
    [trialU, trialV, reprojectInfo] = h2_cleanup_project_velocity(trialU, trialV, repCache, poissonCache, opts, 'projectedRhs');
    info.reprojectInfo = reprojectInfo;
end

evalUc = eval_full_bspf_scalar_derivatives(trialU, repCache);
evalVc = eval_full_bspf_scalar_derivatives(trialV, repCache);
corrNormal = h1_boundary_vector_norm(evalUc.F, evalVc.F);
corrDiv = max(abs(evalUc.Fx + evalVc.Fy), [], 'all');

info.corrNormalBoundaryLinf = corrNormal.linf;
info.corrDivLinf = corrDiv;
info.coeffChangeRel = norm([du; dv]) / baseNorm;
info.scaleLimiter = scale;

divLimit = max(opts.compatCorrectionDivAbsTol, opts.compatCorrectionMaxDivGrowth * max(rawDiv, eps));
accept = (corrNormal.linf <= rawNormal.linf) && (corrDiv <= divLimit);
if accept
    repKuc = trialU;
    repKvc = trialV;
    info.wasAccepted = true;
else
    repKuc = repKu;
    repKvc = repKv;
    info.rejectReason = sprintf('normal %.3e->%.3e, div %.3e->%.3e limit %.3e', ...
        rawNormal.linf, corrNormal.linf, rawDiv, corrDiv, divLimit);
end
end


function pc = h2_a1_normal_boundary_precompute(cache)
Ny = cache.Ny;
Nx = cache.Nx;

tolX = 1.0e-12 * max(1, max(abs(cache.Bx(:))));
tolY = 1.0e-12 * max(1, max(abs(cache.By(:))));
edgeX = find(abs(cache.Bx(:,1)) > tolX | abs(cache.Bx(:,end)) > tolX);
edgeY = find(abs(cache.By(:,1)) > tolY | abs(cache.By(:,end)) > tolY);

uDofs = struct('jx', {}, 'jy', {});
for jy = 1:cache.nbasis
    for jx = edgeX(:).'
        uDofs(end+1) = struct('jx', jx, 'jy', jy); %#ok<AGROW>
    end
end

vDofs = struct('jx', {}, 'jy', {});
for jy = edgeY(:).'
    for jx = 1:cache.nbasis
        vDofs(end+1) = struct('jx', jx, 'jy', jy); %#ok<AGROW>
    end
end

Vu = zeros(2*Ny, numel(uDofs));
for q = 1:numel(uDofs)
    phi = cache.By(uDofs(q).jy,:).' * cache.Bx(uDofs(q).jx,:);
    Vu(:, q) = [phi(:,1); phi(:,end)];
end

Vv = zeros(2*Nx, numel(vDofs));
for q = 1:numel(vDofs)
    phi = cache.By(vDofs(q).jy,:).' * cache.Bx(vDofs(q).jx,:);
    Vv(:, q) = [phi(1,:).'; phi(end,:).'];
end

pc = struct();
pc.Vu = sparse(Vu);
pc.Vv = sparse(Vv);
pc.uDofs = uDofs;
pc.vDofs = vDofs;
end


function bc = h2_velocity_boundary_bc(opts)
bc = [];
if isfield(opts, 'useBoundaryConstrainedDecomposition') && ~opts.useBoundaryConstrainedDecomposition
    return;
end
bc = struct();
bc.value_left = 0;
bc.value_right = 0;
bc.value_bottom = 0;
bc.value_top = 0;
bc.use_kkt_endpoint_constraints = isfield(opts, 'useKktEndpointBoundaryConstraints') && opts.useKktEndpointBoundaryConstraints;
bc.description = 'velocity value no-slip boundary constraint';
end


function [F, info] = h2_apply_scalar_boundary_constraints(F, cache, bc)
F = double(F);
dx = cache.x(2) - cache.x(1);
dy = cache.y(2) - cache.y(1);
[Ny, Nx] = size(F);
info = struct();
info.description = '';
if isfield(bc, 'description')
    info.description = bc.description;
end
info.applied = true;

hasValueLeft = isfield(bc, 'value_left');
hasValueRight = isfield(bc, 'value_right');
hasValueBottom = isfield(bc, 'value_bottom');
hasValueTop = isfield(bc, 'value_top');

if hasValueLeft,   F(:,1) = bc.value_left; end
if hasValueRight,  F(:,end) = bc.value_right; end
if hasValueBottom, F(1,:) = bc.value_bottom; end
if hasValueTop,    F(end,:) = bc.value_top; end

applyD1OnGrid = ~isfield(bc, 'use_kkt_endpoint_constraints') || ~bc.use_kkt_endpoint_constraints;

if applyD1OnGrid && Nx >= 3 && isfield(bc, 'd1_left')
    F(:,2) = (2*dx*bc.d1_left + 3*F(:,1) + F(:,3)) / 4;
end
if applyD1OnGrid && Nx >= 3 && isfield(bc, 'd1_right')
    F(:,end-1) = (3*F(:,end) + F(:,end-2) - 2*dx*bc.d1_right) / 4;
end
if applyD1OnGrid && Ny >= 3 && isfield(bc, 'd1_bottom')
    F(2,:) = (2*dy*bc.d1_bottom + 3*F(1,:) + F(3,:)) / 4;
end
if applyD1OnGrid && Ny >= 3 && isfield(bc, 'd1_top')
    F(end-1,:) = (3*F(end,:) + F(end-2,:) - 2*dy*bc.d1_top) / 4;
end

if hasValueLeft,   F(:,1) = bc.value_left; end
if hasValueRight,  F(:,end) = bc.value_right; end
if hasValueBottom, F(1,:) = bc.value_bottom; end
if hasValueTop,    F(end,:) = bc.value_top; end

info.maxBoundaryValue = max([max(abs(F(:,1))), max(abs(F(:,end))), max(abs(F(1,:))), max(abs(F(end,:)))]);
if Nx >= 3
    dLeft = (-3*F(:,1) + 4*F(:,2) - F(:,3)) / (2*dx);
    dRight = (3*F(:,end) - 4*F(:,end-1) + F(:,end-2)) / (2*dx);
else
    dLeft = 0; dRight = 0;
end
if Ny >= 3
    dBottom = (-3*F(1,:) + 4*F(2,:) - F(3,:)) / (2*dy);
    dTop = (3*F(end,:) - 4*F(end-1,:) + F(end-2,:)) / (2*dy);
else
    dBottom = 0; dTop = 0;
end
info.maxBoundaryD1 = max([max(abs(dLeft(:))), max(abs(dRight(:))), max(abs(dBottom(:))), max(abs(dTop(:)))]);
end


function W = h2_divide_by_noslip_envelope(F, cache)
G = cache.noSlipG;
W = zeros(size(F));
mask = abs(G) > 10*eps(max(1, max(abs(G(:)))));
W(mask) = F(mask) ./ G(mask);
if cache.Nx >= 3
    W(:,1) = W(:,2);
    W(:,end) = W(:,end-1);
end
if cache.Ny >= 3
    W(1,:) = W(2,:);
    W(end,:) = W(end-1,:);
end
end


function E = eval_full_bspf_scalar_derivatives(rep, cache)
if isfield(rep, 'envelope') && strcmp(rep.envelope, 'noslip-bubble')
    baseRep = rep;
    baseRep = rmfield(baseRep, 'envelope');
    W = eval_full_bspf_scalar_derivatives(baseRep, cache);
    G = cache.noSlipG;
    Gx = cache.noSlipGx;
    Gy = cache.noSlipGy;
    LapG = cache.noSlipLapG;
    E = struct();
    E.F = G .* W.F;
    E.Fx = Gx .* W.F + G .* W.Fx;
    E.Fy = Gy .* W.F + G .* W.Fy;
    E.Lap = LapG .* W.F + 2*(Gx .* W.Fx + Gy .* W.Fy) + G .* W.Lap;
    return
end
A1 = rep.A1; A2 = rep.A2; A3 = rep.A3;
Bx = cache.Bx; dBx = cache.dBx; ddBx = cache.ddBx;
By = cache.By; dBy = cache.dBy; ddBy = cache.ddBy;
Nx = cache.Nx; Ny = cache.Ny; Nx0 = cache.Nx0; Ny0 = cache.Ny0;

% BB part.
F_BB   = By.'  * A1.' * Bx;
Fx_BB  = By.'  * A1.' * dBx;
Fy_BB  = dBy.' * A1.' * Bx;
Lap_BB = By.'  * A1.' * ddBx + ddBy.' * A1.' * Bx;

% BF part: By(y) times Fourier-x periodic functions.
F_A2 = zeros(Ny,Nx); Fx_A2 = zeros(Ny,Nx); Fy_A2 = zeros(Ny,Nx); Lap_A2 = zeros(Ny,Nx);
kx = cache.fft.KX(1,:);
for jy = 1:size(A2,1)
    c = A2(jy,:);
    [gx, gxx, g] = periodic_1d_values_derivatives_from_coeff(c, kx);
    gFull = [g, g(1)];
    gxFull = [gx, gx(1)];
    gxxFull = [gxx, gxx(1)];
    F_A2   = F_A2   + By(jy,:).'  * gFull;
    Fx_A2  = Fx_A2  + By(jy,:).'  * gxFull;
    Fy_A2  = Fy_A2  + dBy(jy,:).' * gFull;
    Lap_A2 = Lap_A2 + By(jy,:).'  * gxxFull + ddBy(jy,:).' * gFull;
end

% FB part: Bx(x) times Fourier-y periodic functions.
F_A3 = zeros(Ny,Nx); Fx_A3 = zeros(Ny,Nx); Fy_A3 = zeros(Ny,Nx); Lap_A3 = zeros(Ny,Nx);
ky = cache.fft.KY(:,1).';
for jx = 1:size(A3,1)
    c = A3(jx,:);
    [gy, gyy, g] = periodic_1d_values_derivatives_from_coeff(c, ky);
    gFull = [g(:); g(1)];
    gyFull = [gy(:); gy(1)];
    gyyFull = [gyy(:); gyy(1)];
    F_A3   = F_A3   + gFull   * Bx(jx,:);
    Fx_A3  = Fx_A3  + gFull   * dBx(jx,:);
    Fy_A3  = Fy_A3  + gyFull  * Bx(jx,:);
    Lap_A3 = Lap_A3 + gFull   * ddBx(jx,:) + gyyFull * Bx(jx,:);
end

% Pure periodic residual part.
f0 = rep.f_per(1:Ny0,1:Nx0);
Fhat = fft2(f0);
Fx0 = real(ifft2(1i*cache.fft.KX .* Fhat));
Fy0 = real(ifft2(1i*cache.fft.KY .* Fhat));
Lap0 = real(ifft2(-cache.fft.K2 .* Fhat));
F_per = embed_periodic_full(f0);
Fx_per = embed_periodic_full(Fx0);
Fy_per = embed_periodic_full(Fy0);
Lap_per = embed_periodic_full(Lap0);

E = struct();
E.F   = F_BB   + F_A2   + F_A3   + F_per;
E.Fx  = Fx_BB  + Fx_A2  + Fx_A3  + Fx_per;
E.Fy  = Fy_BB  + Fy_A2  + Fy_A3  + Fy_per;
E.Lap = Lap_BB + Lap_A2 + Lap_A3 + Lap_per;
E.parts = struct('F_BB',F_BB,'F_A2',F_A2,'F_A3',F_A3,'F_per',F_per);
end


function [g1, g2, g0] = periodic_1d_values_derivatives_from_coeff(c, k)
% c is normalized FFT coefficient: c = fft(values)/N.
N = numel(c);
g0 = real(ifft(c*N));
g1 = real(ifft((1i*k(:).') .* (c*N)));
g2 = real(ifft((-(k(:).').^2) .* (c*N)));
end


function w = cmp_high_order_quad_weights_vector(x)
% Robust fallback quadrature weights for uniform/nonuniform 1D nodes.
x = x(:);
n = numel(x);
if n < 2
    w = 1;
    return;
end
w = zeros(n,1);
dx = diff(x);
w(1) = dx(1)/2;
w(end) = dx(end)/2;
if n > 2
    w(2:end-1) = (dx(1:end-1) + dx(2:end))/2;
end
end


function val = weighted_mean_2d(A, x, y)
% Weighted mean over a tensor-product grid using high-order 1D quadrature.
% A is Ny-by-Nx, x is 1-by-Nx or Nx-by-1, y is 1-by-Ny or Ny-by-1.
x = x(:).';
y = y(:).';
wx = cmp_high_order_quad_weights_vector(x);
wy = cmp_high_order_quad_weights_vector(y);
area = (x(end)-x(1)) * (y(end)-y(1));
val = (wy(:).' * A * wx(:)) / max(area, eps);
end


function s = cmp_set_default(s, name, value)
if ~isfield(s, name) || isempty(s.(name))
    s.(name) = value;
end
end


%% ========================================================================
% Cached finite-difference and BSPF-KKT Poisson operators for Stage F
% ========================================================================

function cache = bspf_kkt_poisson_neumann_precompute(x, y, params)
% Precompute everything in the BSPF-KKT Neumann solver that depends only on
% grid, domain, basis parameters, and Trefftz settings.  The time-varying
% data are only f and qL/qR/qB/qT.

x = x(:).';
y = y(:).';
Nx = numel(x);
Ny = numel(y);

p0 = default_bspf_kkt_neumann_params(Nx, Ny);
params = merge_struct_defaults(params, p0);
params.optSplit = merge_struct_defaults(params.optSplit, p0.optSplit);
params.optLapN  = merge_struct_defaults(params.optLapN,  p0.optLapN);

Lx = x(end) - x(1);
Ly = y(end) - y(1);
Nx0 = Nx - 1;
Ny0 = Ny - 1;

degB = params.degB;
nbasis = params.nbasis;
s_newbasis = params.s_newbasis;
delta_newbasis = params.delta_newbasis;
reportSeam = params.reportSeam;
doWaitbar = params.doWaitbar;
optSplit = params.optSplit;
optSplit.degB = degB;
optSplit.showWaitbar = doWaitbar;
optLapN = params.optLapN;

% 1D B-spline values and cached 1D KKT split maps.
[BxSpline, Bvals_x] = make_bspline_basis_values(x, degB, nbasis);
[BySpline, Bvals_y] = make_bspline_basis_values(y, degB, nbasis);
splitX = bspf_kkt_1d_decompose_precompute(x, Bvals_x, optSplit.kmax, optSplit.r, optSplit.lambda_kkt, BxSpline, optSplit);
splitY = bspf_kkt_1d_decompose_precompute(y, Bvals_y, optSplit.kmax, optSplit.r, optSplit.lambda_kkt, BySpline, optSplit);

Bx0 = Bvals_x(:,1:Nx0);
By0 = Bvals_y(:,1:Ny0);
lam = 1e-12;
dx_const = (Bx0*Bx0.' + lam*eye(nbasis)) \ (Bx0*ones(Nx0,1));
dy_const = (By0*By0.' + lam*eye(nbasis)) \ (By0*ones(Ny0,1));

% Non-periodic BB response kernels.
optBB = struct();
optBB.deg                 = degB;
optBB.nbasis              = nbasis;
optBB.s                   = s_newbasis;
optBB.Lx                  = Lx;
optBB.Ly                  = Ly;
optBB.Nx                  = Nx;
optBB.Ny                  = Ny;
optBB.delta               = delta_newbasis;
optBB.report_seam         = reportSeam;
optBB.print_seam_examples = false;
optBB.seam_rel_warn       = 1e-2;
optBB.show_waitbar        = doWaitbar;
kerBB = build_U_basis_new_inline(optBB);

% Mixed B-spline/Fourier modal response kernels.
optExpX = struct();
optExpX.deg          = degB;
optExpX.nbasis       = nbasis;
optExpX.s            = s_newbasis;
optExpX.basis_L      = Lx;
optExpX.mode_L       = Ly;
optExpX.basis_N      = Nx;
optExpX.mode_N       = Ny;
optExpX.delta        = delta_newbasis;
optExpX.n_modes      = floor(Ny/2) - 1;
optExpX.show_waitbar = doWaitbar;
kerExpX = build_newbasis_exp_kernel_1d_inline(optExpX);

optExpY = struct();
optExpY.deg          = degB;
optExpY.nbasis       = nbasis;
optExpY.s            = s_newbasis;
optExpY.basis_L      = Ly;
optExpY.mode_L       = Lx;
optExpY.basis_N      = Ny;
optExpY.mode_N       = Nx;
optExpY.delta        = delta_newbasis;
optExpY.n_modes      = floor(Nx/2) - 1;
optExpY.show_waitbar = doWaitbar;
kerExpY = build_newbasis_exp_kernel_1d_inline(optExpY);

xw = x; xw(end) = xw(1);
yw = y; yw(end) = yw(1);
ell_list_y = kerExpX.ell_list;
ell_list_x = kerExpY.ell_list;
phaseY = zeros(Ny, numel(ell_list_y));
for k = 1:numel(ell_list_y)
    phaseY(:,k) = exp(1i*ell_list_y(k)*yw(:));
end
phaseX = zeros(numel(ell_list_x), Nx);
for k = 1:numel(ell_list_x)
    phaseX(k,:) = exp(1i*ell_list_x(k)*xw(:)).';
end

% FFT wave numbers for the periodic residual solve.
fftCache = spectral_poisson_2d_uniform_precompute(Nx0, Ny0, Lx, Ly);

% Trefftz-Neumann matrix, column scaling, solver matrix, and grid basis.
lapCache = laplace_rect_solver_trefftz_neumann_precompute(x, y, optLapN);

% Boundary quadrature weights for repeated flux checks/corrections.
boundary = struct();
boundary.wx = high_order_quad_weights_vector(x);
boundary.wy = high_order_quad_weights_vector(y);
boundary.measure = 2*sum(boundary.wx) + 2*sum(boundary.wy);

cache = struct();
cache.x = x; cache.y = y;
cache.Nx = Nx; cache.Ny = Ny; cache.Nx0 = Nx0; cache.Ny0 = Ny0;
cache.Lx = Lx; cache.Ly = Ly;
cache.params = params;
cache.optSplit = optSplit;
cache.optLapN = optLapN;
cache.degB = degB;
cache.nbasis = nbasis;
cache.Bvals_x = Bvals_x;
cache.Bvals_y = Bvals_y;
cache.splitX = splitX;
cache.splitY = splitY;
cache.dx_const = dx_const;
cache.dy_const = dy_const;
cache.kerBB = kerBB;
cache.U_BB_mat  = reshape(kerBB.U_basis_new, nbasis*nbasis, Ny*Nx);
cache.qL_BB_mat = reshape(kerBB.qL_BB, nbasis*nbasis, Ny);
cache.qR_BB_mat = reshape(kerBB.qR_BB, nbasis*nbasis, Ny);
cache.qB_BB_mat = reshape(kerBB.qB_BB, nbasis*nbasis, Nx);
cache.qT_BB_mat = reshape(kerBB.qT_BB, nbasis*nbasis, Nx);
cache.Tx = kerBB.Tx;
cache.Ty = kerBB.Ty;
cache.kerExpX = kerExpX;
cache.kerExpY = kerExpY;
cache.Gx_new = kerExpX.G1_new;
cache.Gy_new = kerExpY.G1_new;
cache.Gx_dx_left = kerExpX.G1_d_left;
cache.Gx_dx_right = kerExpX.G1_d_right;
cache.Gy_dy_bottom = kerExpY.G1_d_left;
cache.Gy_dy_top = kerExpY.G1_d_right;
cache.T1x = kerExpX.T;
cache.T1y = kerExpY.T;
cache.ell_list_y = ell_list_y;
cache.ell_list_x = ell_list_x;
cache.phaseY = phaseY;
cache.phaseX = phaseX;
cache.Kpos_x = floor((Nx0-1)/2);
cache.Kpos_y = floor((Ny0-1)/2);
cache.fft = fftCache;
cache.laplace = lapCache;
cache.boundary = boundary;
end


function [U, info] = bspf_kkt_poisson_neumann_apply_cached(f, qL, qR, qB, qT, cache, solution_mean)
% Cached application of the BSPF-KKT Neumann Poisson solver.
% Only f and boundary data vary between calls.

if nargin < 7 || isempty(solution_mean)
    solution_mean = 0;
end

[Ny, Nx] = size(f);
if Ny ~= cache.Ny || Nx ~= cache.Nx
    error('Cached Poisson apply: f size must be [%d,%d].', cache.Ny, cache.Nx);
end

nbasis = cache.nbasis;
Nx0 = cache.Nx0;
Ny0 = cache.Ny0;
Lx = cache.Lx;
Ly = cache.Ly;

% -------------------- 1. Cached BSPF--KKT RHS split --------------------
[f_per, A1_f, A2_f, A3_f, splitDbg] = split2D_kkt_directional_N0_cached(f, cache);

A3_dc = A3_f(:,1);
A2_dc = A2_f(:,1);
A1_f = A1_f + A3_dc * (cache.dy_const.');
A1_f = A1_f + cache.dx_const * (A2_dc.');
A3_f(:,1) = 0;
A2_f(:,1) = 0;

mu_per = mean(f_per(1:Ny0,1:Nx0), 'all');
f0 = f_per(1:Ny0,1:Nx0) - mu_per;
f_per = embed_periodic_full(f0);
A1_f = A1_f + mu_per * (cache.dx_const * (cache.dy_const.'));

% -------------------- 2. Coefficient transforms --------------------
A1_new = cache.Tx \ A1_f / (cache.Ty.');
A3_new = cache.T1x \ A3_f;
A2_new = cache.T1y \ A2_f;

% -------------------- 3. Assemble particular solution and dUp/dn --------
A1vec = A1_new(:);
U_total = reshape((A1vec.' * cache.U_BB_mat), Ny, Nx);
qP_L = (A1vec.' * cache.qL_BB_mat).';
qP_R = (A1vec.' * cache.qR_BB_mat).';
qP_B =  A1vec.' * cache.qB_BB_mat;
qP_T =  A1vec.' * cache.qT_BB_mat;

Kx_use = min([numel(cache.ell_list_x), cache.Kpos_x, size(A2_new,2)-1]);
Ky_use = min([numel(cache.ell_list_y), cache.Kpos_y, size(A3_new,2)-1]);

% B_x x exp(i ell y) part.
for iK = 1:Ky_use
    ell    = cache.ell_list_y(iK);
    ky_idx = iK + 1;
    phase_y = cache.phaseY(:,iK);

    Gx_slice = squeeze(cache.Gx_new(iK, :, :));
    s_x = (A3_new(:,ky_idx).') * Gx_slice;
    U_total = U_total + 2*real(phase_y * s_x);

    sx_left_dx  = (A3_new(:,ky_idx).') * (cache.Gx_dx_left(iK,:).');
    sx_right_dx = (A3_new(:,ky_idx).') * (cache.Gx_dx_right(iK,:).');

    qP_L = qP_L - 2*real(phase_y * sx_left_dx);
    qP_R = qP_R + 2*real(phase_y * sx_right_dx);
    qP_B = qP_B - 2*real(1i*ell*phase_y(1)   * s_x);
    qP_T = qP_T + 2*real(1i*ell*phase_y(end) * s_x);
end

% B_y x exp(i ell x) part.
for iK = 1:Kx_use
    ell    = cache.ell_list_x(iK);
    kx_idx = iK + 1;
    phase_x = cache.phaseX(iK,:);

    Gy_slice = squeeze(cache.Gy_new(iK, :, :));
    s_y = (A2_new(:,kx_idx).') * Gy_slice;
    U_total = U_total + 2*real((s_y.') * phase_x);

    qP_L = qP_L - 2*real((s_y.') * (1i*ell*phase_x(1)));
    qP_R = qP_R + 2*real((s_y.') * (1i*ell*phase_x(end)));

    sy_bottom_dy = (A2_new(:,kx_idx).') * (cache.Gy_dy_bottom(iK,:).');
    sy_top_dy    = (A2_new(:,kx_idx).') * (cache.Gy_dy_top(iK,:).');

    qP_B = qP_B - 2*real(sy_bottom_dy * phase_x);
    qP_T = qP_T + 2*real(sy_top_dy    * phase_x);
end

% Periodic FFT particular solution and spectral normal derivatives.
f0 = f_per(1:Ny0,1:Nx0);
[Phi0, Phi0_x, Phi0_y] = spectral_poisson_2d_uniform_with_grad_cached(f0, cache.fft);
Phi   = embed_periodic_full(Phi0);
Phi_x = embed_periodic_full(Phi0_x);
Phi_y = embed_periodic_full(Phi0_y);

U_particular = U_total + Phi;
qP_L = qP_L - Phi_x(:,1);
qP_R = qP_R + Phi_x(:,end);
qP_B = qP_B - Phi_y(1,:);
qP_T = qP_T + Phi_y(end,:);

% -------------------- 4. Cached Laplace-Neumann correction --------------
qL_corr = qL(:)   - qP_L(:);
qR_corr = qR(:)   - qP_R(:);
qB_corr = qB(:).' - qP_B(:).';
qT_corr = qT(:).' - qP_T(:).';

flux_before = boundary_flux_rect_cached(qL_corr, qR_corr, qB_corr, qT_corr, cache.boundary);
flux_after = flux_before;
flux_correction_applied = false;

if isfield(cache.optLapN, 'flux_correction_tol') && ~isempty(cache.optLapN.flux_correction_tol)
    flux_tol = cache.optLapN.flux_correction_tol;
else
    flux_tol = 0;
end

if isfield(cache.optLapN, 'remove_flux_mean') && cache.optLapN.remove_flux_mean && abs(flux_before) > flux_tol
    corr = flux_before / cache.boundary.measure;
    qL_corr = qL_corr(:)   - corr;
    qR_corr = qR_corr(:)   - corr;
    qB_corr = qB_corr(:).' - corr;
    qT_corr = qT_corr(:).' - corr;
    flux_after = boundary_flux_rect_cached(qL_corr, qR_corr, qB_corr, qT_corr, cache.boundary);
    flux_correction_applied = true;
end

[U_lap, lapDiag] = laplace_rect_solver_trefftz_neumann_apply_cached( ...
    qL_corr, qR_corr, qB_corr, qT_corr, cache.laplace);

U_raw = U_particular + U_lap;
U = U_raw - mean(U_raw(:)) + solution_mean;

info = struct();
info.U_particular = U_particular;
info.U_lap = U_lap;
info.qP_L = qP_L;
info.qP_R = qP_R;
info.qP_B = qP_B;
info.qP_T = qP_T;
info.qL_corr = qL_corr;
info.qR_corr = qR_corr;
info.qB_corr = qB_corr;
info.qT_corr = qT_corr;
info.flux_before = flux_before;
info.flux_after = flux_after;
info.flux_correction_applied = flux_correction_applied;
info.split = splitDbg;
info.laplace = lapDiag;
info.params = cache.params;
end


function pc = bspf_kkt_1d_decompose_precompute(t, Bvals, kmax, r, lambda_kkt, Bspline, opt)
if nargin < 6
    Bspline = [];
end
if nargin < 7 || isempty(opt)
    opt = struct();
end
if ~isfield(opt, 'endpointDerivativeMethod') || isempty(opt.endpointDerivativeMethod)
    opt.endpointDerivativeMethod = 'fd';
end
if ~isfield(opt, 'endpointDerivativeRadius') || isempty(opt.endpointDerivativeRadius)
    opt.endpointDerivativeRadius = r;
end
if ~isfield(opt, 'endpointDerivativeDegree') || isempty(opt.endpointDerivativeDegree)
    opt.endpointDerivativeDegree = kmax + 1;
end
t = t(:).';
N = numel(t);
nb = size(Bvals,1);
Bmat = Bvals.';                 % N x nb
w = trapezoid_weights_1d(t);     % N x 1
sw = sqrt(w(:));
Bw = Bmat .* sw;
H = Bw.' * Bw + lambda_kkt * eye(nb);

[E0, E1] = endpoint_matrix_for_basis(Bvals, t, kmax, r, Bspline);
E = zeros(2*(kmax+1), nb);
for m = 1:(kmax+1)
    E(2*m-1, :) = E0(m, :);
    E(2*m,   :) = E1(m, :);
end

D0 = endpoint_derivative_matrix_for_grid(N, t, kmax, opt.endpointDerivativeRadius, "left", opt.endpointDerivativeMethod, opt.endpointDerivativeDegree);
D1 = endpoint_derivative_matrix_for_grid(N, t, kmax, opt.endpointDerivativeRadius, "right", opt.endpointDerivativeMethod, opt.endpointDerivativeDegree);
Dpair = zeros(2*(kmax+1), N);
for m = 1:(kmax+1)
    Dpair(2*m-1,:) = D0(m,:);
    Dpair(2*m,:)   = D1(m,:);
end

KKT = [H, E.'; E, zeros(size(E,1))];
rhsMap = Bmat.' .* (w(:).');
rhsFullMap = [rhsMap; Dpair];
if rcond(KKT) < 1e-14
    solveMap = pinv(KKT) * rhsFullMap;
    KKTdec = [];
else
    solveMap = KKT \ rhsFullMap;
    KKTdec = decomposition(KKT, 'lu');
end

pc = struct();
pc.Bmat = Bmat;
pc.cMap = solveMap(1:nb,:);
pc.KKT = KKT;
pc.KKTdec = KKTdec;
pc.rhsMap = rhsMap;
pc.Dpair = Dpair;
pc.N = N;
pc.nb = nb;
pc.kmax = kmax;
pc.r = r;
pc.lambda_kkt = lambda_kkt;
pc.KKT_rcond = rcond(KKT);
pc.endpointDerivativeMethod = opt.endpointDerivativeMethod;
pc.endpointDerivativeRadius = opt.endpointDerivativeRadius;
pc.endpointDerivativeDegree = opt.endpointDerivativeDegree;
if isempty(Bspline)
    pc.endpoint_basis_derivative = 'finite-difference';
else
    pc.endpoint_basis_derivative = 'analytic-bspline';
end
end


function [f_nonper, c, f_per] = bspf_kkt_1d_decompose_apply_cached(v, pc, bc)
if nargin < 3
    bc = [];
end
v = v(:);
if numel(v) ~= pc.N
    error('Cached 1D BSPF-KKT split: vector length mismatch.');
end
if isempty(bc)
    c = pc.cMap * v;
else
    rhs = [pc.rhsMap * v; pc.Dpair * v];
    rhs = h2_apply_1d_endpoint_constraint_rhs(rhs, pc, bc);
    if isempty(pc.KKTdec)
        sol = pinv(pc.KKT) * rhs;
    else
        sol = pc.KKTdec \ rhs;
    end
    c = sol(1:pc.nb);
end
f_nonper = pc.Bmat * c;
f_per = v - f_nonper;
edge = 0.5 * (f_per(1) + f_per(end));
f_per(1) = edge;
f_per(end) = edge;
end


function rhs = h2_apply_1d_endpoint_constraint_rhs(rhs, pc, bc)
nEndpoint = 2 * (pc.kmax + 1);
firstEndpointRow = numel(rhs) - nEndpoint + 1;
row = @(m, side) firstEndpointRow + 2*(m - 1) + side - 1;
if isfield(bc, 'value_left')
    rhs(row(1,1)) = bc.value_left;
end
if isfield(bc, 'value_right')
    rhs(row(1,2)) = bc.value_right;
end
if pc.kmax >= 1
    if isfield(bc, 'd1_left')
        rhs(row(2,1)) = bc.d1_left;
    end
    if isfield(bc, 'd1_right')
        rhs(row(2,2)) = bc.d1_right;
    end
end
end


function D = endpoint_derivative_matrix_for_grid(N, t, kmax, r, which_end, method, degree)
if nargin < 6 || isempty(method)
    method = 'fd';
end
if nargin < 7 || isempty(degree)
    degree = kmax + 1;
end
t = t(:).';
dt = t(2)-t(1);
D = zeros(kmax+1, N);
if which_end == "left"
    xi = 0:2*r;
    idx = 1 + xi;
    D(1,1) = 1;
else
    xi = 0:-1:-2*r;
    idx = N + xi;
    D(1,N) = 1;
end
for m = 1:kmax
    if strcmpi(method, 'local-poly-qr')
        a = local_poly_endpoint_coeffs(xi, m, degree) / dt^m;
    else
        a = fd_coeffs_local(xi, m) / dt^m;
    end
    D(m+1,idx) = a(:).';
end
end


function a = local_poly_endpoint_coeffs(xi, m, degree)
xi = xi(:).';
n = numel(xi);
degree = min(max(degree, m), n - 1);
V = zeros(n, degree + 1);
for p = 0:degree
    V(:, p + 1) = xi(:).^p;
end
rhs = zeros(degree + 1, 1);
rhs(m + 1) = factorial(m);
% If c = pinv(V) * y are local polynomial coefficients, derivative at the
% endpoint is rhs' * c, so the direct stencil is rhs' * pinv(V).
a = (rhs.' * pinv(V)).';
end


function [f_per, A1, A2, A3, dbg] = split2D_kkt_directional_N0_cached(f, cache, bc)
if nargin < 3
    bc = [];
end
[Ny, Nx] = size(f);
Nx0 = cache.Nx0;
Ny0 = cache.Ny0;
nbx = cache.nbasis;
nby = cache.nbasis;
bcY = h2_directional_endpoint_bc(bc, 'y');
bcX = h2_directional_endpoint_bc(bc, 'x');

Cy = zeros(nby, Nx);
Fy = zeros(Ny, Nx);
Ry = zeros(Ny, Nx);
for ix = 1:Nx
    [fy_np, cy, fy_per] = bspf_kkt_1d_decompose_apply_cached(f(:,ix), cache.splitY, bcY);
    Cy(:, ix) = cy(:);
    Fy(:, ix) = fy_np(:);
    Ry(:, ix) = fy_per(:);
end

A1 = zeros(nbx, nby);
A2 = zeros(nby, Nx0);
Cy_x_nonper = zeros(nby, Nx);
Cy_x_per    = zeros(nby, Nx);
Cxy_coef    = zeros(nbx, nby);
for jy = 1:nby
    [cx_np, cxy, cx_per] = bspf_kkt_1d_decompose_apply_cached(Cy(jy,:).', cache.splitX);
    Cy_x_nonper(jy,:) = cx_np(:).';
    Cy_x_per(jy,:)    = cx_per(:).';
    Cxy_coef(:,jy)    = cxy(:);
    A1(:,jy) = A1(:,jy) + cxy(:);
    tmp = cx_per(:).';
    tmp(end) = tmp(1);
    A2(jy,:) = fft(tmp(1:Nx0)) / Nx0;
end

Cx = zeros(nbx, Ny);
Fx = zeros(Ny, Nx);
Rxy = zeros(Ny, Nx);
for iy = 1:Ny
    [fx_np, cx, fx_per] = bspf_kkt_1d_decompose_apply_cached(Ry(iy,:).', cache.splitX, bcX);
    Cx(:,iy) = cx(:);
    Fx(iy,:) = fx_np(:).';
    Rxy(iy,:) = fx_per(:).';
end

A3 = zeros(nbx, Ny0);
Cx_y_nonper = zeros(nbx, Ny);
Cx_y_per    = zeros(nbx, Ny);
Cyx_coef    = zeros(nby, nbx);
for ixb = 1:nbx
    [cy_np, cyx, cy_per] = bspf_kkt_1d_decompose_apply_cached(Cx(ixb,:).', cache.splitY);
    Cx_y_nonper(ixb,:) = cy_np(:).';
    Cx_y_per(ixb,:)    = cy_per(:).';
    Cyx_coef(:,ixb)    = cyx(:);
    A1(ixb,:) = A1(ixb,:) + cyx(:).';
    tmp = cy_per(:).';
    tmp(end) = tmp(1);
    A3(ixb,:) = fft(tmp(1:Ny0)) / Ny0;
end

f_per = Rxy;
f_per(Ny, 1:Nx0) = f_per(1, 1:Nx0);
f_per(1:Ny0, Nx) = f_per(1:Ny0, 1);
f_per(Ny, Nx)    = f_per(1, 1);

dbg = struct();
dbg.method = 'cached directional BSPF-KKT, no DeltaY/DeltaX split';
dbg.lambda_kkt = cache.optSplit.lambda_kkt;
dbg.endpointBoundaryConstraints = ~isempty(bcX) || ~isempty(bcY);
dbg.Cy = Cy;
dbg.Cx = Cx;
dbg.Fy = Fy;
dbg.Fx = Fx;
dbg.Ry = Ry;
dbg.Rxy = Rxy;
dbg.Cy_x_nonper = Cy_x_nonper;
dbg.Cy_x_per = Cy_x_per;
dbg.Cx_y_nonper = Cx_y_nonper;
dbg.Cx_y_per = Cx_y_per;
dbg.Cxy_coef = Cxy_coef;
dbg.Cyx_coef = Cyx_coef;
try
    f_nonper_recon = reconstruct_nonper_from_A123_N0(A1, A2, A3, cache.Bvals_x, cache.Bvals_y);
    dbg.reconstruction_linf = max(abs(f_nonper_recon(:) + f_per(:) - f(:)));
catch
    dbg.reconstruction_linf = NaN;
end
end


function bc1 = h2_directional_endpoint_bc(bc, direction)
bc1 = [];
if isempty(bc) || ~isfield(bc, 'use_kkt_endpoint_constraints') || ~bc.use_kkt_endpoint_constraints
    return;
end
bc1 = struct();
if strcmpi(direction, 'x')
    if isfield(bc, 'value_left'),  bc1.value_left  = bc.value_left;  end
    if isfield(bc, 'value_right'), bc1.value_right = bc.value_right; end
    if isfield(bc, 'd1_left'),     bc1.d1_left     = bc.d1_left;     end
    if isfield(bc, 'd1_right'),    bc1.d1_right    = bc.d1_right;    end
else
    if isfield(bc, 'value_bottom'), bc1.value_left  = bc.value_bottom; end
    if isfield(bc, 'value_top'),    bc1.value_right = bc.value_top;    end
    if isfield(bc, 'd1_bottom'),    bc1.d1_left     = bc.d1_bottom;    end
    if isfield(bc, 'd1_top'),       bc1.d1_right    = bc.d1_top;       end
end
if isempty(fieldnames(bc1))
    bc1 = [];
end
end


function fftCache = spectral_poisson_2d_uniform_precompute(Nx, Ny, Lx, Ly)
kx_int = [0:floor(Nx/2), -ceil(Nx/2)+1:-1];
ky_int = [0:floor(Ny/2), -ceil(Ny/2)+1:-1];
kx = (2*pi/Lx) * kx_int;
ky = (2*pi/Ly) * ky_int;
[KX, KY] = meshgrid(kx, ky);
K2 = KX.^2 + KY.^2;
fftCache = struct('Nx',Nx,'Ny',Ny,'KX',KX,'KY',KY,'K2',K2,'mask',K2~=0);
end


function [Phi, Phix, Phiy] = spectral_poisson_2d_uniform_with_grad_cached(f, fftCache)
f = double(f);
[Ny, Nx] = size(f);
if Ny ~= fftCache.Ny || Nx ~= fftCache.Nx
    error('Cached spectral Poisson: input size mismatch.');
end
Fhat = fft2(f);
Phi_hat = zeros(Ny, Nx);
Phi_hat(fftCache.mask) = -Fhat(fftCache.mask) ./ fftCache.K2(fftCache.mask);
Phi_hat(1,1) = 0;
Phi  = real(ifft2(Phi_hat));
Phix = real(ifft2(1i*fftCache.KX .* Phi_hat));
Phiy = real(ifft2(1i*fftCache.KY .* Phi_hat));
end


function lapCache = laplace_rect_solver_trefftz_neumann_precompute(x, y, opt)
if nargin < 3 || isempty(opt)
    opt = struct();
end
if ~isfield(opt, 'K') || isempty(opt.K), opt.K = 24; end
if ~isfield(opt, 'lambda') || isempty(opt.lambda), opt.lambda = 0; end
if ~isfield(opt, 'zero_mean') || isempty(opt.zero_mean), opt.zero_mean = true; end

K = opt.K;
lambda = opt.lambda;
x = x(:).';
y = y(:).';
Nx = numel(x);
Ny = numel(y);
xmin = x(1); xmax = x(end); ymin = y(1); ymax = y(end);
xc = 0.5*(xmin+xmax);
yc = 0.5*(ymin+ymax);
R = 0.5*sqrt((xmax-xmin)^2 + (ymax-ymin)^2);

XL = xmin * ones(Ny,1); YL = y(:); nXL = -ones(Ny,1); nYL = zeros(Ny,1);
XR = xmax * ones(Ny,1); YR = y(:); nXR =  ones(Ny,1); nYR = zeros(Ny,1);
XB = x(:); YB = ymin * ones(Nx,1); nXB = zeros(Nx,1); nYB = -ones(Nx,1);
XT = x(:); YT = ymax * ones(Nx,1); nXT = zeros(Nx,1); nYT =  ones(Nx,1);
Xbd = [XL; XR; XB; XT];
Ybd = [YL; YR; YB; YT];
nx = [nXL; nXR; nXB; nXT];
ny = [nYL; nYR; nYB; nYT];

wy = trapezoid_weights_vector(y);
wx = trapezoid_weights_vector(x);
wbd = [wy(:); wy(:); wx(:); wx(:)];
sw = sqrt(wbd(:));

M = numel(Xbd);
nbasis = 2*K;
A = zeros(M, nbasis);
zeta = ((Xbd-xc) + 1i*(Ybd-yc))/R;
for k = 1:K
    dzdx = k * zeta.^(k-1) / R;
    dzdy = 1i * k * zeta.^(k-1) / R;
    dRe_dn = nx .* real(dzdx) + ny .* real(dzdy);
    dIm_dn = nx .* imag(dzdx) + ny .* imag(dzdy);
    A(:,2*k-1) = dRe_dn;
    A(:,2*k)   = dIm_dn;
end

Aw = A .* sw;
colScale = sqrt(sum(abs(Aw).^2,1));
colScale(colScale < eps) = 1;
As = Aw ./ colScale;
if lambda > 0
    solverMat = (As.'*As + lambda*eye(nbasis)) \ (As.');
else
    solverMat = pinv(As);
end

[Xg, Yg] = meshgrid(x, y);
zeta_grid = ((Xg-xc) + 1i*(Yg-yc))/R;
Hgrid = zeros(Nx*Ny, nbasis);
for k = 1:K
    zk = zeta_grid.^k;
    Hgrid(:,2*k-1) = real(zk(:));
    Hgrid(:,2*k)   = imag(zk(:));
end

lapCache = struct();
lapCache.method = 'cached Trefftz-Neumann';
lapCache.K = K;
lapCache.lambda = lambda;
lapCache.zero_mean = opt.zero_mean;
lapCache.Nx = Nx; lapCache.Ny = Ny;
lapCache.A = A;
lapCache.sw = sw;
lapCache.colScale = colScale(:);
lapCache.As = As;
lapCache.solverMat = solverMat;
lapCache.Hgrid = Hgrid;
lapCache.cond_scaled = cond(As);
end


function [U, diag] = laplace_rect_solver_trefftz_neumann_apply_cached(q0v, q1v, h0v, h1v, lapCache)
qbd = [q0v(:); q1v(:); h0v(:); h1v(:)];
cs = lapCache.solverMat * (qbd .* lapCache.sw);
coef = cs(:) ./ lapCache.colScale(:);
qfit = lapCache.A * coef;
bdry_res = qfit - qbd;
U = reshape(lapCache.Hgrid * coef, lapCache.Ny, lapCache.Nx);
if lapCache.zero_mean
    U = U - mean(U(:));
end
diag = struct();
diag.method = lapCache.method;
diag.K = lapCache.K;
diag.nbasis = 2*lapCache.K;
diag.coef = coef;
diag.bdry_res_linf = max(abs(bdry_res));
diag.bdry_res_l2 = sqrt(mean(bdry_res.^2));
diag.cond_scaled = lapCache.cond_scaled;
diag.qbd = qbd;
diag.qfit = qfit;
end


function flux = boundary_flux_rect_cached(qL, qR, qB, qT, boundaryCache)
flux = sum(boundaryCache.wy(:) .* qL(:)) ...
     + sum(boundaryCache.wy(:) .* qR(:)) ...
     + sum(boundaryCache.wx(:) .* qB(:)) ...
     + sum(boundaryCache.wx(:) .* qT(:));
end


%% ========================================================================
%% Integrated local copy of the user's BSPF-KKT Neumann Poisson solver
%% ------------------------------------------------------------------------
%% The following functions are included so this file is self-contained.
%% No separate bspf_kkt_poisson_neumann_clean.m file is required.
%% ========================================================================


function params = default_bspf_kkt_neumann_params(Nx, Ny)
% Default parameters for the clean Neumann BSPF--KKT solver.

if nargin < 1, Nx = 100; end
if nargin < 2, Ny = Nx; end

params = struct();
params.verbose = false;
params.doWaitbar = false;

params.optSplit = struct();
params.optSplit.kmax = 5;
params.optSplit.r = 3;
params.optSplit.lambda_kkt = 1e-10;
params.optSplit.showWaitbar = false;

params.degB = 6;
params.nbasis = 18;
params.s_newbasis = params.optSplit.kmax;
params.delta_newbasis = 0.60;
params.reportSeam = false;

params.optLapN = struct();
params.optLapN.K = min(44, floor(Nx/2));
params.optLapN.lambda = 1e-14;
params.optLapN.remove_flux_mean = true;
params.optLapN.zero_mean = true;
params.optLapN.flux_correction_tol = 1e-9;

params.solution_mean = 0;
end


function out = merge_struct_defaults(in, defaults)
out = defaults;
if isempty(in)
    return;
end
names = fieldnames(in);
for k = 1:numel(names)
    out.(names{k}) = in.(names{k});
end
end


function [Ux, Uy] = periodic_gradient_2d_real(U, Px, Py)
% Real-valued FFT spectral gradient on the extended periodic domain; used for Neumann boundary derivative kernels.
[Ny, Nx] = size(U);

kx_int = [0:floor(Nx/2), -ceil(Nx/2)+1:-1];
ky_int = [0:floor(Ny/2), -ceil(Ny/2)+1:-1];

kx = (2*pi/Px) * kx_int;
ky = (2*pi/Py) * ky_int;

[KX, KY] = meshgrid(kx, ky);

Uhat = fft2(U);

Ux = real(ifft2(1i*KX .* Uhat));
Uy = real(ifft2(1i*KY .* Uhat));
end


function [Ux, Uy] = periodic_gradient_2d_complex(U, Px, Py)
% Complex-valued FFT spectral gradient on the extended periodic domain; used to extract derivatives of B-spline times exp(i*ell*y) modal kernels.
[Ny, Nx] = size(U);

kx_int = [0:floor(Nx/2), -ceil(Nx/2)+1:-1];
ky_int = [0:floor(Ny/2), -ceil(Ny/2)+1:-1];

kx = (2*pi/Px) * kx_int;
ky = (2*pi/Py) * ky_int;

[KX, KY] = meshgrid(kx, ky);

Uhat = fft2(U);

Ux = ifft2(1i*KX .* Uhat);
Uy = ifft2(1i*KY .* Uhat);
end


function w = trapezoid_weights_vector(x)
% Keep the original function name, but use high-order boundary quadrature weights internally.
% Uniform grids use composite Simpson 1/3 plus Simpson 3/8 weights; nonuniform grids fall back to trapezoidal weights.
w = high_order_quad_weights_vector(x);
end


function w = high_order_quad_weights_vector(x)
% High-order 1D quadrature weights on a uniform endpoint grid.
x = x(:).';
N = numel(x);
if N < 2
    error('high_order_quad_weights_vector: N must be >= 2.');
end

hvec = diff(x);
h = mean(hvec);
if max(abs(hvec - h)) > 100*eps(max(1, abs(h)))
    w = zeros(1, N);
    w(1) = hvec(1)/2;
    w(end) = hvec(end)/2;
    if N > 2
        w(2:end-1) = 0.5*(hvec(1:end-1) + hvec(2:end));
    end
    return;
end

m = N - 1;
w = zeros(1, N);
if m == 1
    w(:) = h/2;
    return;
end

if mod(m, 2) == 0
    w(1) = h/3;
    w(end) = h/3;
    w(2:2:end-1) = 4*h/3;
    w(3:2:end-2) = 2*h/3;
else
    mSim = m - 3;
    if mSim > 0
        idxEnd = mSim + 1;
        w(1) = w(1) + h/3;
        w(idxEnd) = w(idxEnd) + h/3;
        if idxEnd >= 3
            w(2:2:idxEnd-1) = w(2:2:idxEnd-1) + 4*h/3;
            w(3:2:idxEnd-2) = w(3:2:idxEnd-2) + 2*h/3;
        end
    else
        idxEnd = 1;
    end
    ids = idxEnd:(idxEnd+3);
    w(ids) = w(ids) + (3*h/8) * [1, 3, 3, 1];
end
end


%% ========================================================================
%% Original solver local functions retained from the uploaded version
%% ========================================================================


function w = trapezoid_weights_1d(t)
% One-dimensional trapezoidal quadrature weights.
    t = t(:);
    N = numel(t);
    if N < 2
        error('trapezoid_weights_1d: at least two grid points are required.');
    end
    w = zeros(N,1);
    dt = diff(t);
    w(1) = dt(1) / 2;
    w(end) = dt(end) / 2;
    if N > 2
        w(2:end-1) = 0.5 * (dt(1:end-1) + dt(2:end));
    end
end


function [B, Bvals, knots] = make_bspline_basis_values(t, deg, nbasis)
% Build only B-spline basis objects and their grid values; do not construct interpolation splines q_m.
% This is the primitive B-spline basis used in the BSPF-KKT decomposition.
    if nargin < 2, deg = 6; end
    if nargin < 3, nbasis = 18; end

    t = t(:).';
    t0 = t(1);
    t1 = t(end);
    L = t1 - t0;
    if L <= 0
        error('make_bspline_basis_values: t must be strictly increasing.');
    end

    tau = (t - t0) / L;
    order = deg + 1;
    nseg = nbasis - deg;
    if nseg < 1
        error('deg=%d too large for nbasis=%d (need nbasis-deg >= 1).', deg, nbasis);
    end

    knots_unit = [zeros(1, order), (1:nseg-1)/nseg, ones(1, order)];
    coefsB = eye(nbasis);
    B = spmak(knots_unit, coefsB);
    Bvals = fnval(B, tau);
    knots = t0 + L * knots_unit;
end


function [A0, A1] = endpoint_matrix_for_basis(Bvals, t, kmax, r, Bspline)
    if nargin < 5
        Bspline = [];
    end
    t = t(:).'; N = numel(t); dt = t(2)-t(1); nb = size(Bvals,1);
    A0 = zeros(kmax+1, nb); A1 = zeros(kmax+1, nb);
    A0(1,:) = Bvals(:,1).'; A1(1,:) = Bvals(:,end).';
    if ~isempty(Bspline)
        L = t(end) - t(1);
        for m = 1:kmax
            dB = fnder(Bspline, m);
            A0(m+1,:) = fnval(dB, 0).' / L^m;
            A1(m+1,:) = fnval(dB, 1).' / L^m;
        end
        return
    end
    xi0 = 0:2*r; idx0 = 1 + xi0;
    xi1 = 0:-1:-2*r; idx1 = N + xi1;
    for m = 1:kmax
        a0 = fd_coeffs_local(xi0, m);
        a1 = fd_coeffs_local(xi1, m);
        for j = 1:nb
            bj = Bvals(j,:);
            A0(m+1,j) = dot(a0(:), bj(idx0).') / dt^m;
            A1(m+1,j) = dot(a1(:), bj(idx1).') / dt^m;
        end
    end
end


function a = fd_coeffs_local(xi, m)
    xi = xi(:).';
    n = numel(xi);
    V = zeros(n,n);
    for k = 0:n-1
        V(k+1,:) = xi.^k;
    end
    rhs = zeros(n,1);
    rhs(m+1) = factorial(m);
    a = V \ rhs;
end


function f_nonper = reconstruct_nonper_from_A123_N0(A1, A2, A3, Bx, By)
    % A2: nby-by-Nx0, A3: nbx-by-Ny0
    [nby, Nx0] = size(A2);
    [nbx, Ny0] = size(A3);

    Ny = size(By,2); Nx = size(Bx,2);

    % BB
    fBB = zeros(Ny,Nx);
    for jx = 1:nbx
        for jy = 1:nby
            fBB = fBB + A1(jx,jy) * (By(jy,:).') * (Bx(jx,:));
        end
    end

    % A2: By(y) * exp(i*kx*x)
    fA2 = zeros(Ny,Nx);
    for jy = 1:nby
        tmpx0 = ifft(A2(jy,:)*Nx0, 'symmetric'); % 1-by-Nx0
        tmpx  = zeros(1,Nx);
        tmpx(1:Nx0) = tmpx0;
        tmpx(end)   = tmpx0(1);
        fA2 = fA2 + (By(jy,:).') * tmpx;
    end

    % A3: Bx(x) * exp(i*ky*y)
    fA3 = zeros(Ny,Nx);
    for jx = 1:nbx
        tmpy0 = ifft(A3(jx,:)*Ny0, 'symmetric'); % 1-by-Ny0
        tmpy  = zeros(Ny,1);
        tmpy(1:Ny0) = tmpy0(:);
        tmpy(end)   = tmpy0(1);
        fA3 = fA3 + tmpy * (Bx(jx,:));
    end

    f_nonper = fBB + fA2 + fA3;
end


function Afull = embed_periodic_full(A0)
% A0: unique periodic block (Ny0-by-Nx0) -> full block (Ny0+1-by-Nx0+1) by copying endpoints.
    [Ny0,Nx0] = size(A0);
    Afull = zeros(Ny0+1, Nx0+1);
    Afull(1:Ny0,1:Nx0) = A0;
    Afull(end,1:Nx0)   = A0(1,:);
    Afull(1:Ny0,end)   = A0(:,1);
    Afull(end,end)     = A0(1,1);
end


function ker = build_U_basis_new_inline(opt)
% Rewritten from the original document-2 routine: return U_basis_new / Tx / Ty directly without saving a MAT file.

    if nargin < 1
        opt = struct();
    end

    opt = set_default(opt, 'deg', 6);
    opt = set_default(opt, 'nbasis', 14);
    opt = set_default(opt, 's', 3);
    opt = set_default(opt, 'Lx', 1.0);
    opt = set_default(opt, 'Ly', 1.0);
    opt = set_default(opt, 'Nx', 32);
    opt = set_default(opt, 'Ny', 32);
    opt = set_default(opt, 'delta', 0.60);
    opt = set_default(opt, 'report_seam', false);
    opt = set_default(opt, 'print_seam_examples', false);
    opt = set_default(opt, 'seam_rel_warn', 1e-2);
    opt = set_default(opt, 'show_waitbar', false);

    deg    = opt.deg;
    nbasis = opt.nbasis;
    s      = opt.s;
    Lx     = opt.Lx;
    Ly     = opt.Ly;
    Nx     = opt.Nx;
    Ny     = opt.Ny;
    delta  = opt.delta;

    fprintf('  -> building U_basis_new: deg=%d, nbasis=%d, s=%d, Nx=%d, Ny=%d, delta=%.3f\n', ...
        deg, nbasis, s, Nx, Ny, delta);

    if s > deg
        error('Require s <= deg.');
    end
    if nbasis < 2*(deg+1)
        error('Need nbasis >= 2*(deg+1).');
    end
    if Nx < 2 || Ny < 2
        error('Nx and Ny must be at least 2.');
    end

    x = linspace(0, Lx, Nx);
    y = linspace(0, Ly, Ny);

    hx = Lx / (Nx - 1);
    hy = Ly / (Ny - 1);

    Nx_pad = round(delta / hx);
    Ny_pad = round(delta / hy);

    delta_x_eff = Nx_pad * hx;
    delta_y_eff = Ny_pad * hy;

    Nx_ext = Nx + 2*Nx_pad;
    Ny_ext = Ny + 2*Ny_pad;

    xext = (-Nx_pad : Nx-1+Nx_pad) * hx;
    yext = (-Ny_pad : Ny-1+Ny_pad) * hy;

    ix0 = Nx_pad + (1:Nx);
    iy0 = Ny_pad + (1:Ny);

    Lx_box = xext(end) - xext(1);
    Ly_box = yext(end) - yext(1);

    Px = Nx_ext * hx;
    Py = Ny_ext * hy;

    [Xext_grid, Yext_grid] = meshgrid(xext, yext);
    Qx_unit = 0.5 * (Xext_grid - (xext(1) + 0.5*Px));
    Qy_unit = 0.5 * (Yext_grid - (yext(1) + 0.5*Py));

    order = deg + 1;
    nsegx = nbasis - deg;
    nsegy = nbasis - deg;

    t_interior_x = (1:nsegx-1) / nsegx;
    t_interior_y = (1:nsegy-1) / nsegy;

    knots_x = [zeros(1, order), Lx*t_interior_x, Lx*ones(1, order)];
    knots_y = [zeros(1, order), Ly*t_interior_y, Ly*ones(1, order)];

    modesX = build_1d_modes_symmetric_two_sided_nozm(knots_x, nbasis, deg, s, xext);
    modesY = build_1d_modes_symmetric_two_sided_nozm(knots_y, nbasis, deg, s, yext);

    Tx = modesX.T;
    Ty = modesY.T;
    metaX = modesX.meta;
    metaY = modesY.meta;

    U_basis_new = zeros(nbasis, nbasis, Ny, Nx);

    % Neumann translation: store only the normal-derivative kernels on the four physical boundaries.
    % These derivatives are computed by FFT spectral differentiation on the extended periodic domain, not by one-sided differences after cropping.
    qL_BB = zeros(nbasis, nbasis, Ny);
    qR_BB = zeros(nbasis, nbasis, Ny);
    qB_BB = zeros(nbasis, nbasis, Nx);
    qT_BB = zeros(nbasis, nbasis, Nx);

    F_mean_cont = zeros(nbasis, nbasis);
    F_mean_disc = zeros(nbasis, nbasis);
    F_mean      = zeros(nbasis, nbasis);

    seam_x_abs = zeros(nbasis, nbasis);
    seam_y_abs = zeros(nbasis, nbasis);
    seam_x_rel = zeros(nbasis, nbasis);
    seam_y_rel = zeros(nbasis, nbasis);

    Ntotal = nbasis * nbasis;
    counter = 0;
    tStart = tic;
    isCanceled = false;

    if opt.show_waitbar
        hwb = waitbar(0, 'Building U_basis_new inline ...', ...
            'Name', 'inline U\_basis\_new', ...
            'CreateCancelBtn', 'setappdata(gcbf,''canceling'',true)');
        setappdata(hwb, 'canceling', false);
    else
        hwb = [];
    end

    try
        for ixb = 1:nbasis
            px = modesX.Phi_ext(ixb,:);

            for iyb = 1:nbasis
                drawnow;
                if ~isempty(hwb)
                    if ~ishandle(hwb) || getappdata(hwb, 'canceling')
                        isCanceled = true;
                        break;
                    end
                end

                py = modesY.Phi_ext(iyb,:);
                Fext = py(:) * px(:).';

                mean_cont = trapz(yext, trapz(xext, Fext, 2)) / (Lx_box * Ly_box);
                mean_disc = mean(Fext(:));

                F_mean_cont(ixb, iyb) = mean_cont;
                F_mean_disc(ixb, iyb) = mean_disc;
                F_mean(ixb, iyb)      = mean_disc;

                sxa = max(abs(Fext(:,1) - Fext(:,end)));
                sya = max(abs(Fext(1,:) - Fext(end,:)));
                fmax = max(abs(Fext(:)));

                seam_x_abs(ixb, iyb) = sxa;
                seam_y_abs(ixb, iyb) = sya;
                seam_x_rel(ixb, iyb) = sxa / (fmax + eps);
                seam_y_rel(ixb, iyb) = sya / (fmax + eps);

                if opt.print_seam_examples
                    if seam_x_rel(ixb,iyb) > opt.seam_rel_warn || seam_y_rel(ixb,iyb) > opt.seam_rel_warn
                        fprintf('[seam warn] pair (%d,%d): seam_x_abs=%.3e, seam_y_abs=%.3e, seam_x_rel=%.3e, seam_y_rel=%.3e\n', ...
                            ixb, iyb, sxa, sya, seam_x_rel(ixb,iyb), seam_y_rel(ixb,iyb));
                    end
                end

                Feff = Fext - mean_disc;
                Vext = poisson_fft_periodic_2d_zero_mean(Feff, Px, Py);
                Qext = build_quadratic_particular(xext, yext, hx, hy, mean_disc);
                Uext = Vext + Qext;

                U_basis_new(ixb, iyb, :, :) = Uext(iy0, ix0);

                [Vext_x, Vext_y] = periodic_gradient_2d_real(Vext, Px, Py);
                Uext_x = Vext_x + mean_disc * Qx_unit;
                Uext_y = Vext_y + mean_disc * Qy_unit;

                qL_BB(ixb, iyb, :) = -Uext_x(iy0, ix0(1));
                qR_BB(ixb, iyb, :) =  Uext_x(iy0, ix0(end));
                qB_BB(ixb, iyb, :) = -Uext_y(iy0(1), ix0);
                qT_BB(ixb, iyb, :) =  Uext_y(iy0(end), ix0);

                counter = counter + 1;
                if ~isempty(hwb)
                    if mod(counter, max(1, floor(Ntotal/100))) == 0 || counter == Ntotal
                        update_waitbar_core(hwb, counter, Ntotal, tStart, ...
                            sprintf('basis pair (%d,%d) / (%d,%d)', ixb, iyb, nbasis, nbasis));
                    end
                elseif mod(counter, max(1, floor(Ntotal/10))) == 0 || counter == Ntotal
                    fprintf('    U_basis_new progress: %d / %d (%.1f%%)\n', counter, Ntotal, 100*counter/Ntotal);
                end
            end

            if isCanceled
                break;
            end
        end

        if ~isempty(hwb) && ishandle(hwb)
            delete(hwb);
        end

    catch ME
        if ~isempty(hwb) && ishandle(hwb)
            delete(hwb);
        end
        rethrow(ME);
    end

    if isCanceled
        error('U_basis_new construction was canceled.');
    end

    if opt.report_seam
        [mx_sx, idx_sx] = max(seam_x_rel(:));
        [mx_sy, idx_sy] = max(seam_y_rel(:));
        [ix_sx, iy_sx] = ind2sub(size(seam_x_rel), idx_sx);
        [ix_sy, iy_sy] = ind2sub(size(seam_y_rel), idx_sy);

        fprintf('  seam_x_rel max = %.3e at pair (%d,%d)\n', mx_sx, ix_sx, iy_sx);
        fprintf('  seam_y_rel max = %.3e at pair (%d,%d)\n', mx_sy, ix_sy, iy_sy);
    end

    fprintf('  -> U_basis_new inline completed: size = [%d, %d, %d, %d]\n', ...
        size(U_basis_new,1), size(U_basis_new,2), size(U_basis_new,3), size(U_basis_new,4));

    ker = struct();
    ker.U_basis_new = U_basis_new;
    ker.qL_BB = qL_BB;
    ker.qR_BB = qR_BB;
    ker.qB_BB = qB_BB;
    ker.qT_BB = qT_BB;
    ker.Tx = Tx;
    ker.Ty = Ty;
    ker.metaX = metaX;
    ker.metaY = metaY;
    ker.x = x;
    ker.y = y;
    ker.xext = xext;
    ker.yext = yext;
    ker.ix0 = ix0;
    ker.iy0 = iy0;
    ker.F_mean = F_mean;
    ker.F_mean_cont = F_mean_cont;
    ker.F_mean_disc = F_mean_disc;
    ker.seam_x_abs = seam_x_abs;
    ker.seam_y_abs = seam_y_abs;
    ker.seam_x_rel = seam_x_rel;
    ker.seam_y_rel = seam_y_rel;
    ker.hx = hx;
    ker.hy = hy;
    ker.Px = Px;
    ker.Py = Py;
    ker.delta_x_eff = delta_x_eff;
    ker.delta_y_eff = delta_y_eff;
end


function ker = build_newbasis_exp_kernel_1d_inline(opt)
% Inline precomputation for the document-3 two-dimensional modal Poisson kernel.
% For each (ell, basis) pair, solve the 2D subproblem directly:
%       Delta U = phi_ext(x) * exp(i*ell*y)
% Then extract the modal amplitude g(x) and store it as G1_new(iL,ib,:).
%
% Compatibility with the legacy main-program interface:
%   - basis_L / basis_N: length and grid size in the non-periodic basis direction.
%   - mode_L  / mode_N : length and full grid size in the Fourier periodic direction.
%   - N remains available as a legacy alias for basis_N; quad_n is ignored.

    if nargin < 1
        opt = struct();
    end

    opt = set_default(opt, 'deg', 6);
    opt = set_default(opt, 'nbasis', 14);
    opt = set_default(opt, 's', 3);

    if isfield(opt, 'basis_L') && ~isempty(opt.basis_L)
        basis_L_default = opt.basis_L;
    else
        basis_L_default = 1.0;
    end
    opt = set_default(opt, 'basis_L', basis_L_default);
    opt = set_default(opt, 'mode_L', opt.basis_L);

    if isfield(opt, 'basis_N') && ~isempty(opt.basis_N)
        basis_N_default = opt.basis_N;
    elseif isfield(opt, 'N') && ~isempty(opt.N)
        basis_N_default = opt.N;
    else
        basis_N_default = 32;
    end
    opt = set_default(opt, 'basis_N', basis_N_default);
    opt = set_default(opt, 'N', opt.basis_N);

    if isfield(opt, 'mode_N') && ~isempty(opt.mode_N)
        mode_N_default = opt.mode_N;
    else
        mode_N_default = opt.basis_N;
    end
    opt = set_default(opt, 'mode_N', mode_N_default);

    opt = set_default(opt, 'delta', 0.60);
    opt = set_default(opt, 'n_modes', floor((opt.mode_N - 2)/2));
    opt = set_default(opt, 'show_waitbar', false);

    deg     = opt.deg;
    nbasis  = opt.nbasis;
    s       = opt.s;
    basis_L = opt.basis_L;
    mode_L  = opt.mode_L;
    basis_N = opt.basis_N;
    mode_N  = opt.mode_N;
    delta   = opt.delta;
    n_modes = opt.n_modes;

    if s > deg
        error('Require s <= deg.');
    end
    if nbasis < 2*(deg+1)
        error('Need nbasis >= 2*(deg+1).');
    end
    if basis_N < 2
        error('basis_N must be at least 2.');
    end
    if mode_N < 3
        error('mode_N must be at least 3.');
    end
    if n_modes < 0
        error('n_modes cannot be negative.');
    end

    fprintf('  -> building 2D modal Poisson kernel: basis(L,N)=(%.6g,%d), mode(L,N)=(%.6g,%d), n_modes=%d, delta=%.3f\n', ...
        basis_L, basis_N, mode_L, mode_N, n_modes, delta);

    % Non-periodic direction: original interval plus two-sided padding.
    x = linspace(0, basis_L, basis_N);
    hx = basis_L / (basis_N - 1);

    Nx_pad = round(delta / hx);
    delta_eff = Nx_pad * hx;
    xext = (-Nx_pad : basis_N-1+Nx_pad) * hx;
    Nx_ext = numel(xext);
    ix0 = Nx_pad + (1:basis_N);

    % Periodic direction: unique periodic grid used for FFT modes.
    Ny0 = mode_N - 1;
    hy = mode_L / Ny0;
    y0 = (0:Ny0-1) * hy;

    % FFT periodic lengths matched to the discrete grids.
    Px = Nx_ext * hx;
    Py = mode_L;

    order = deg + 1;
    nseg  = nbasis - deg;
    t_interior = (1:nseg-1) / nseg;
    knots = [zeros(1, order), basis_L*t_interior, basis_L*ones(1, order)];

    modes = build_1d_modes_symmetric_two_sided_nozm(knots, nbasis, deg, s, xext);

    T     = modes.T;
    meta  = modes.meta;
    check = modes.check;
    Phi_ext = modes.Phi_ext;

    ell_list = (2*pi/mode_L) * (1:n_modes);

    G1_new = zeros(n_modes, nbasis, basis_N);

    % Endpoint derivative kernels: compute FFT spectral derivatives of Uxy on the extended periodic domain, then extract modal-amplitude derivatives.
    G1_d_left  = zeros(n_modes, nbasis);
    G1_d_right = zeros(n_modes, nbasis);

    modal_res_rel = zeros(n_modes, nbasis);
    modal_imag_amp = zeros(n_modes, nbasis);

    Ntotal = max(1, n_modes * nbasis);
    counter = 0;
    tStart = tic;
    isCanceled = false;

    if opt.show_waitbar
        hwb = waitbar(0, 'Building G1_new (2D modal Poisson) ...', ...
            'Name', 'inline 2D modal Poisson kernel', ...
            'CreateCancelBtn', 'setappdata(gcbf,''canceling'',true)');
        setappdata(hwb, 'canceling', false);
    else
        hwb = [];
    end

    try
        for iL = 1:n_modes
            ell = ell_list(iL);
            phase_y = exp(1i * ell * y0(:));

            for ib = 1:nbasis
                drawnow;
                if ~isempty(hwb)
                    if ~ishandle(hwb) || getappdata(hwb, 'canceling')
                        isCanceled = true;
                        break;
                    end
                end

                phi_x = Phi_ext(ib, :);
                Fxy = phase_y * phi_x;

                Uxy = poisson_fft_periodic_2d_zero_mean_complex(Fxy, Px, Py);

                [Uxy_x, ~] = periodic_gradient_2d_complex(Uxy, Px, Py);

                phase_rep = repmat(phase_y, 1, Nx_ext);
                amp_x    = mean(Uxy   ./ phase_rep, 1);
                amp_x_dx = mean(Uxy_x ./ phase_rep, 1);

                Urec = phase_y * amp_x;
                modal_res_rel(iL, ib) = max(abs(Uxy(:) - Urec(:))) / (max(abs(Uxy(:))) + eps);
                modal_imag_amp(iL, ib) = max(abs(imag(amp_x)));

                G1_new(iL, ib, :) = real(amp_x(ix0));
                G1_d_left(iL, ib)  = real(amp_x_dx(ix0(1)));
                G1_d_right(iL, ib) = real(amp_x_dx(ix0(end)));

                counter = counter + 1;
                if ~isempty(hwb)
                    if mod(counter, max(1, floor(Ntotal/200))) == 0 || counter == Ntotal
                        update_waitbar_core(hwb, counter, Ntotal, tStart, ...
                            sprintf('mode %d/%d, basis %d/%d', iL, n_modes, ib, nbasis));
                    end
                elseif mod(counter, max(1, floor(Ntotal/10))) == 0 || counter == Ntotal
                    fprintf('    G1_new progress: %d / %d (%.1f%%%%)\n', counter, Ntotal, 100*counter/Ntotal);
                end
            end

            if isCanceled
                break;
            end
        end

        if ~isempty(hwb) && ishandle(hwb)
            delete(hwb);
        end

    catch ME
        if ~isempty(hwb) && ishandle(hwb)
            delete(hwb);
        end
        rethrow(ME);
    end

    if isCanceled
        error('G1_new construction was canceled.');
    end

    fprintf('  -> 2D modal Poisson kernel inline completed: size(G1_new) = [%d, %d, %d]\n', ...
        size(G1_new,1), size(G1_new,2), size(G1_new,3));
    fprintf('     max modal_res_rel = %.3e, max imag(amp_x) = %.3e\n', ...
        max(modal_res_rel(:)), max(modal_imag_amp(:)));

    ker = struct();
    ker.G1_new = G1_new;
    ker.G1_d_left = G1_d_left;
    ker.G1_d_right = G1_d_right;
    ker.ell_list = ell_list;
    ker.T = T;
    ker.meta = meta;
    ker.x = x;
    ker.xext = xext;
    ker.ix0 = ix0;
    ker.y0 = y0;
    ker.knots = knots;
    ker.check = check;
    ker.opt = opt;
    ker.hx = hx;
    ker.hy = hy;
    ker.Px = Px;
    ker.Py = Py;
    ker.delta_eff = delta_eff;
    ker.modal_res_rel = modal_res_rel;
    ker.modal_imag_amp = modal_imag_amp;
end


function U = poisson_fft_periodic_2d_zero_mean_complex(F, Px, Py)
[Ny, Nx] = size(F);

    Fhat = fft2(F);
    Fhat(1,1) = 0;

    kx = (2*pi/Px) * [0:floor(Nx/2), -floor((Nx-1)/2):-1];
    ky = (2*pi/Py) * [0:floor(Ny/2), -floor((Ny-1)/2):-1];
    [KX, KY] = meshgrid(kx, ky);
    K2 = KX.^2 + KY.^2;

    Uhat = zeros(Ny, Nx);
    mask = (K2 > 0);
    Uhat(mask) = -Fhat(mask) ./ K2(mask);
    Uhat(~mask) = 0;

    U = ifft2(Uhat);
end


function modes = build_1d_modes_symmetric_two_sided_nozm(knots, nbasis, deg, s, xext)
    if s > deg
        error('Require s <= deg.');
    end

    tol = 1e-12;
    order = deg + 1;
    L = knots(end);

    sp1D = cell(1, nbasis);
    supp = zeros(nbasis, 2);
    for j = 1:nbasis
        coefs = zeros(1, nbasis);
        coefs(j) = 1;
        sp1D{j} = spmak(knots, coefs);
        supp(j,1) = knots(j);
        supp(j,2) = knots(j + order);
    end

    idxL = find(abs(supp(:,1) - knots(1)) < tol);
    idxR = find(abs(supp(:,2) - knots(end)) < tol);
    idxI = setdiff(1:nbasis, union(idxL, idxR));

    overlapLR = intersect(idxL, idxR);
    if ~isempty(overlapLR)
        error('Left/right boundary spaces overlap: %s', mat2str(overlapLR));
    end
    if numel(idxL) ~= numel(idxR)
        error('Left/right boundary DOF counts differ: nL=%d, nR=%d.', numel(idxL), numel(idxR));
    end

    DL = build_boundary_derivative_matrix(sp1D, idxL, knots(1), s);
    if rank(DL, 1e-10) < s+1
        error('Left boundary derivative matrix rank is insufficient.');
    end

    [CjetL, NnullL] = jet_and_null_basis_square(DL);
    [T, meta] = assemble_square_basis_from_left_reflection( ...
        nbasis, idxL, idxI, idxR, CjetL, NnullL, s);

    nmodes = size(T,2);
    Phi_ext = zeros(nmodes, numel(xext));
    for k = 1:nmodes
        Phi_ext(k,:) = extend_mode_two_sided(sp1D, T(:,k), xext, L, meta(k));
    end

    check = validate_stage1_two_sided(sp1D, T, meta, s, L);

    modes = struct();
    modes.knots = knots;
    modes.nbasis = nbasis;
    modes.deg = deg;
    modes.s = s;
    modes.L = L;
    modes.xext = xext;
    modes.sp1D = sp1D;
    modes.supp = supp;
    modes.idxL = idxL;
    modes.idxI = idxI;
    modes.idxR = idxR;
    modes.DL = DL;
    modes.CjetL = CjetL;
    modes.NnullL = NnullL;
    modes.T = T;
    modes.meta = meta;
    modes.Phi_ext = Phi_ext;
    modes.Phi = Phi_ext;
    modes.Phi_zm = Phi_ext;
    modes.check = check;
end


function D = build_boundary_derivative_matrix(sp1D, idx, xbd, s)
    n = numel(idx);
    D = zeros(s+1, n);
    for r = 0:s
        for a = 1:n
            spd = fnder(sp1D{idx(a)}, r);
            D(r+1, a) = fnval(spd, xbd);
        end
    end
end


function [Cjet, Nnull] = jet_and_null_basis_square(D)
    nB = size(D,2);
    r  = size(D,1);

    row_scale = max(abs(D), [], 2);
    row_scale(row_scale == 0) = 1;
    Dscaled = D ./ row_scale;

    Cjet = zeros(nB, r);
    for k = 1:r
        rhs = zeros(r,1);
        rhs(k) = 1 / row_scale(k);
        try
            Cjet(:,k) = lsqminnorm(Dscaled, rhs);
        catch
            Cjet(:,k) = pinv(Dscaled) * rhs;
        end
    end

    try
        Nnull = null(D, 'r');
    catch
        Nnull = null(D);
    end
    if isempty(Nnull)
        Nnull = zeros(nB, 0);
    end
end


function [T, meta] = assemble_square_basis_from_left_reflection(nbasis, idxL, idxI, idxR, CjetL, NnullL, s)
    nL = numel(idxL);
    nR = numel(idxR);

    if nL ~= nR
        error('Left/right boundary DOF counts differ.');
    end

    T = zeros(nbasis, nbasis);
    meta = struct('type',{},'side',{},'r',{},'name',{});
    col = 0;

    leftJetCols = zeros(s+1,1);
    for r = 0:s
        col = col + 1;
        c = zeros(nbasis,1);
        c(idxL) = CjetL(:,r+1);
        T(:,col) = c;
        meta(col) = struct('type','jet','side','L','r',r,'name',sprintf('Ljet%d',r));
        leftJetCols(r+1) = col;
    end

    nNullL = size(NnullL,2);
    leftNullCols = zeros(nNullL,1);
    for k = 1:nNullL
        col = col + 1;
        c = zeros(nbasis,1);
        c(idxL) = NnullL(:,k);
        T(:,col) = c;
        meta(col) = struct('type','null','side','L','r',-1,'name',sprintf('Lnull%d',k));
        leftNullCols(k) = col;
    end

    for k = 1:numel(idxI)
        col = col + 1;
        T(idxI(k), col) = 1;
        meta(col) = struct('type','int','side','I','r',-1,'name',sprintf('I%d',idxI(k)));
    end

    for r = 0:s
        col = col + 1;
        cL = T(:, leftJetCols(r+1));
        T(:,col) = reflect_left_mode_to_right(cL, idxL, idxR, (-1)^r);
        meta(col) = struct('type','jet','side','R','r',r,'name',sprintf('Rjet%d',r));
    end

    for k = 1:nNullL
        col = col + 1;
        cL = T(:, leftNullCols(k));
        T(:,col) = reflect_left_mode_to_right(cL, idxL, idxR, 1.0);
        meta(col) = struct('type','null','side','R','r',-1,'name',sprintf('Rnull%d',k));
    end

    if col ~= nbasis
        error('assemble_square_basis_from_left_reflection: final column count is not nbasis.');
    end
end


function cR = reflect_left_mode_to_right(cL, idxL, idxR, scale)
    localL = cL(idxL);
    localR = scale * flipud(localL(:));
    cR = zeros(size(cL));
    cR(idxR) = localR;
end


function phi = eval_mode_on_interval(sp1D, c, x)
    phi = zeros(size(x));
    for j = 1:numel(c)
        if abs(c(j)) > 0
            phi = phi + c(j) * fnval(sp1D{j}, x);
        end
    end
end


function val = eval_mode_derivative_at(sp1D, c, x0, r)
    val = 0;
    for j = 1:numel(c)
        if abs(c(j)) > 0
            spd = fnder(sp1D{j}, r);
            val = val + c(j) * fnval(spd, x0);
        end
    end
end


function phi_ext = extend_mode_two_sided(sp1D, c, xext, L, meta)
    phi_ext = zeros(size(xext));

    maskIn = (xext >= 0 & xext <= L);
    phi_ext(maskIn) = eval_mode_on_interval(sp1D, c, xext(maskIn));

    maskL = (xext < 0);
    if strcmp(meta.type,'jet') && strcmp(meta.side,'L')
        xr = -xext(maskL);
        phi_ext(maskL) = ((-1)^meta.r) * eval_mode_on_interval(sp1D, c, xr);
    else
        phi_ext(maskL) = 0;
    end

    maskR = (xext > L);
    if strcmp(meta.type,'jet') && strcmp(meta.side,'R')
        xr = 2*L - xext(maskR);
        phi_ext(maskR) = ((-1)^meta.r) * eval_mode_on_interval(sp1D, c, xr);
    else
        phi_ext(maskR) = 0;
    end
end


function check = validate_stage1_two_sided(sp1D, T, meta, s, L)
    check = struct();

    leftJetIds   = find(arrayfun(@(m) strcmp(m.type,'jet')  && strcmp(m.side,'L'), meta));
    leftNullIds  = find(arrayfun(@(m) strcmp(m.type,'null') && strcmp(m.side,'L'), meta));
    rightJetIds  = find(arrayfun(@(m) strcmp(m.type,'jet')  && strcmp(m.side,'R'), meta));
    rightNullIds = find(arrayfun(@(m) strcmp(m.type,'null') && strcmp(m.side,'R'), meta));

    EjetL = zeros(numel(leftJetIds), s+1);
    for q = 1:numel(leftJetIds)
        k = leftJetIds(q);
        c = T(:,k);
        rr = meta(k).r;
        for m = 0:s
            val = eval_mode_derivative_at(sp1D, c, 0.0, m);
            EjetL(q,m+1) = abs(val - double(m == rr));
        end
    end

    EnullL = zeros(numel(leftNullIds), s+1);
    for q = 1:numel(leftNullIds)
        k = leftNullIds(q);
        c = T(:,k);
        for m = 0:s
            EnullL(q,m+1) = abs(eval_mode_derivative_at(sp1D, c, 0.0, m));
        end
    end

    EjetR = zeros(numel(rightJetIds), s+1);
    for q = 1:numel(rightJetIds)
        k = rightJetIds(q);
        c = T(:,k);
        rr = meta(k).r;
        for m = 0:s
            val = eval_mode_derivative_at(sp1D, c, L, m);
            EjetR(q,m+1) = abs(val - double(m == rr));
        end
    end

    EnullR = zeros(numel(rightNullIds), s+1);
    for q = 1:numel(rightNullIds)
        k = rightNullIds(q);
        c = T(:,k);
        for m = 0:s
            EnullR(q,m+1) = abs(eval_mode_derivative_at(sp1D, c, L, m));
        end
    end

    check.left_jet_err = EjetL;
    check.left_null_err = EnullL;
    check.right_jet_err = EjetR;
    check.right_null_err = EnullR;

    check.max_left_jet_err = safe_max(EjetL);
    check.max_left_null_err = safe_max(EnullL);
    check.max_right_jet_err = safe_max(EjetR);
    check.max_right_null_err = safe_max(EnullR);
end


function m = safe_max(A)
    if isempty(A)
        m = 0;
    else
        m = max(A(:), [], 'omitnan');
    end
end


function V = poisson_fft_periodic_2d_zero_mean(F, Px, Py)
    [Ny, Nx] = size(F);

    Fhat = fft2(F);
    Fhat(1,1) = 0;

    kx = (2*pi/Px) * [0:floor(Nx/2), -floor((Nx-1)/2):-1];
    ky = (2*pi/Py) * [0:floor(Ny/2), -floor((Ny-1)/2):-1];
    [KX, KY] = meshgrid(kx, ky);
    K2 = KX.^2 + KY.^2;

    Vhat = zeros(size(Fhat));
    mask = (K2 > 0);
    Vhat(mask) = -Fhat(mask) ./ K2(mask);
    Vhat(~mask) = 0;

    V = real(ifft2(Vhat));
end


function Q = build_quadratic_particular(xext, yext, hx, hy, c0)
    Nx = numel(xext);
    Ny = numel(yext);
    Px = Nx * hx;
    Py = Ny * hy;

    [X, Y] = meshgrid(xext, yext);
    xc = xext(1) + 0.5 * Px;
    yc = yext(1) + 0.5 * Py;

    Q = (c0 / 4) * ( (X - xc).^2 + (Y - yc).^2 );
    Q = Q - mean(Q(:));
end


function update_waitbar_core(hwb, counter, Ntotal, tStart, labelstr)
    if ~ishandle(hwb)
        return;
    end

    frac = counter / Ntotal;
    elapsed = toc(tStart);
    if frac > 0
        eta = elapsed * (1-frac) / frac;
    else
        eta = NaN;
    end

    msg = sprintf(['%d / %d (%.1f%%%%)\n' ...
                   'Elapsed: %.1fs\nETA: %.1fs\n' ...
                   '%s'], ...
                   counter, Ntotal, 100*frac, elapsed, eta, labelstr);

    waitbar(frac, hwb, msg);
    drawnow;
end


function opt = set_default(opt, name, val)
    if ~isfield(opt, name) || isempty(opt.(name))
        opt.(name) = val;
    end
end

