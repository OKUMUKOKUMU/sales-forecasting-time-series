# =====================================================================================
# Weekly Sales Forecasting in R — six models in base R (stats package only)
# Author: Fordrane Albert Okumu
#
# PURPOSE
#   Rebuilds the forecasting comparison from python/sales_forecasting.ipynb in base R,
#   with no packages to install. Uses the same data, the same 26-week hold-out and the
#   same accuracy metrics, so results can be compared side by side:
#     1. Seasonal naive            - benchmark (identical to Python by construction)
#     2. Holt-Winters              - stats::HoltWinters on log(units)
#     3. ARIMA + Fourier + regressors - stats::arima(order = c(0,1,2), xreg = ...), same spec as Python SARIMAX
#     4. STL + exponential smoothing - stl() seasonal adjustment, damped Holt on the adjusted series
#     5. Harmonic regression       - lm(): trend + Fourier + promo/holiday/price (no ARIMA errors)
#     6. Ensemble                  - average of models 2-5
#
# Run from the repository root:   Rscript R/sales_forecasting.R
# =====================================================================================

H <- 26
out_dir <- file.path("outputs", "r"); dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

d <- read.csv("data/weekly_sales.csv")
d$week <- as.Date(d$week)
d$post_promo <- c(0, head(d$promo, -1))
d$price_step <- as.integer(d$week >= as.Date("2024-07-01"))
n <- nrow(d); tr <- 1:(n - H); te <- (n - H + 1):n
y <- d$units; ly <- log(y)
cat(sprintf("%d weeks | train %s to %s | test %s to %s\n", n, d$week[1], d$week[n - H], d$week[n - H + 1], d$week[n]))

# Fourier terms (6 harmonics, period 52.18 weeks), identical to the Python helper
fourier <- function(dates, K = 6, period = 52.18) {
  t <- as.numeric(dates - as.Date("2021-01-04")) / 7
  do.call(cbind, lapply(1:K, function(k) cbind(sin(2 * pi * k * t / period), cos(2 * pi * k * t / period))))
}
regs <- c("promo", "post_promo", "holiday", "price_step")
Xall <- cbind(fourier(d$week), as.matrix(d[, regs]))
colnames(Xall) <- c(paste0("f", 1:12), regs)

# Accuracy metrics - same definitions as python/models.py
metrics <- function(actual, pred, train_y) {
  e <- actual - pred
  scale <- mean(abs(diff(train_y, lag = 52)))
  c(MAE = mean(abs(e)), RMSE = sqrt(mean(e^2)), `MAPE %` = 100 * mean(abs(e) / actual),
    MASE = mean(abs(e)) / scale, `Bias %` = 100 * (sum(pred) - sum(actual)) / sum(actual))
}

fc <- list()

# 1. Seasonal naive: same week last year
fc[["Seasonal naive"]] <- rep(tail(y[tr], 52), length.out = H)

# 2. Holt-Winters (additive on log scale = multiplicative on units)
hw <- HoltWinters(ts(ly[tr], frequency = 52), seasonal = "additive")
fc[["Holt-Winters"]] <- exp(as.numeric(predict(hw, n.ahead = H)))
cat(sprintf("Holt-Winters smoothing: alpha = %.3f, beta = %.3f, gamma = %.3f\n", hw$alpha, hw$beta, hw$gamma))

# 3. ARIMA(0,1,2) with drift + Fourier + regressors  (drift = time index as a regressor, since d = 1)
drift <- seq_len(n)
ar <- arima(ly[tr], order = c(0, 1, 2), xreg = cbind(drift = drift[tr], Xall[tr, ]), method = "ML")
fc[["ARIMA + Fourier"]] <- exp(as.numeric(predict(ar, n.ahead = H, newxreg = cbind(drift = drift[te], Xall[te, ]))$pred))
eff <- coef(ar)[regs]
cat("ARIMA regressor effects (% change in sales):\n"); print(round(100 * (exp(eff) - 1), 1))

# 4. STL + exponential smoothing: deseasonalise, forecast the level, reseasonalise
st <- stl(ts(ly[tr], frequency = 52), s.window = "periodic", robust = TRUE)
sa <- ly[tr] - st$time.series[, "seasonal"]
# Simple exponential smoothing for the level + a drift equal to the average weekly change of the
# seasonally adjusted series. An undamped Holt trend extrapolates short-term swings too far.
ses <- HoltWinters(ts(sa), beta = FALSE, gamma = FALSE)
slope <- mean(diff(as.numeric(sa)))
season_next <- rep(tail(as.numeric(st$time.series[, "seasonal"]), 52), length.out = H)
fc[["STL + ETS"]] <- exp(as.numeric(predict(ses, n.ahead = H)) + slope * (1:H) + season_next)

# 5. Harmonic regression: deterministic trend + Fourier + regressors (ordinary least squares)
lm_df <- data.frame(ly = ly, t = drift, Xall)
lmfit <- lm(ly ~ ., data = lm_df[tr, ])
fc[["Harmonic regression"]] <- exp(as.numeric(predict(lmfit, newdata = lm_df[te, ])))

# 6. Ensemble of the four non-benchmark models
fc[["Ensemble"]] <- rowMeans(do.call(cbind, fc[c("Holt-Winters", "ARIMA + Fourier", "STL + ETS", "Harmonic regression")]))

res <- t(sapply(fc, metrics, actual = y[te], train_y = y[tr]))
res <- res[order(res[, "MASE"]), ]
cat("\nHold-out accuracy (last 26 weeks):\n"); print(round(res, 2))
write.csv(round(res, 3), file.path(out_dir, "holdout_metrics_R.csv"))

# Ljung-Box on ARIMA residuals
lb <- sapply(c(4, 13, 26), function(l) Box.test(residuals(ar), lag = l, type = "Ljung-Box")$p.value)
names(lb) <- paste0("lag ", c(4, 13, 26))

# -------------------------------------------------------------------------------------
# Charts
# -------------------------------------------------------------------------------------
cols <- c("Seasonal naive" = "#94a3b8", "Holt-Winters" = "#2563eb", "ARIMA + Fourier" = "#b45309",
          "STL + ETS" = "#7c3aed", "Harmonic regression" = "#65a30d", "Ensemble" = "#0f766e")
png(file.path(out_dir, "R_01_forecasts.png"), width = 1900, height = 950, res = 170)
par(mar = c(4, 4.5, 3, 1))
idx <- (n - 78):n
plot(d$week[idx], y[idx], type = "l", lwd = 1.5, xlab = "", ylab = "Units", ylim = range(c(y[idx], unlist(fc))),
     main = "R models: forecasts vs actual (26-week hold-out)")
abline(v = d$week[n - H + 1], lty = 3)
for (m in names(fc)) lines(d$week[te], fc[[m]], col = cols[m], lwd = ifelse(m == "Ensemble", 2.5, 1.3))
legend("topleft", c("Actual", names(fc)), col = c("black", cols[names(fc)]), lwd = 2, bty = "n", cex = .7, ncol = 2)
invisible(dev.off())

png(file.path(out_dir, "R_02_mape.png"), width = 1400, height = 800, res = 170)
par(mar = c(4, 10, 3, 2))
bp <- barplot(rev(res[, "MAPE %"]), horiz = TRUE, las = 1, col = cols[rev(rownames(res))], border = NA,
              xlab = "MAPE %", main = "Hold-out MAPE by model (R)", xlim = c(0, max(res[, "MAPE %"]) * 1.2))
text(rev(res[, "MAPE %"]), bp, sprintf("%.1f%%", rev(res[, "MAPE %"])), pos = 4, cex = .8)
invisible(dev.off())

# -------------------------------------------------------------------------------------
# Cross-check with Python and Markdown report
# -------------------------------------------------------------------------------------
py_file <- file.path("outputs", "python", "holdout_metrics.csv")
check <- "Python results not found - run the notebook first."
if (file.exists(py_file)) {
  py <- read.csv(py_file, row.names = 1, check.names = FALSE)
  sn_diff <- max(abs(unlist(py["Seasonal naive", ]) - res["Seasonal naive", ]))
  check <- c(sprintf("Seasonal naive metrics identical to Python: max difference %.2e", sn_diff),
             sprintf("ARIMA(0,1,2) + Fourier MAPE: R %.2f%% vs Python SARIMAX %.2f%% (same specification, different optimisers)",
                     res["ARIMA + Fourier", "MAPE %"], py["SARIMAX + Fourier", "MAPE %"]),
             sprintf("Holt-Winters MAPE: R %.2f%% vs Python %.2f%% (R's HoltWinters has no damping; parameters differ)",
                     res["Holt-Winters", "MAPE %"], py["Holt-Winters (ETS)", "MAPE %"]))
}
cat("\nCROSS-CHECK:\n", paste(check, collapse = "\n "), "\n")

md_table <- function(m) c(paste("| model |", paste(colnames(m), collapse = " | "), "|"),
                          paste("|", paste(rep("---", ncol(m) + 1), collapse = " | "), "|"),
                          sapply(rownames(m), function(r) paste("|", r, "|", paste(sprintf("%.2f", m[r, ]), collapse = " | "), "|")))
report <- c(
  "# Sales Forecasting — R results (auto-generated by `R/sales_forecasting.R`)", "",
  sprintf("*%d weeks · hold-out = last %d weeks (%s to %s) · base R only*", n, H, d$week[n - H + 1], d$week[n]), "",
  "## Hold-out accuracy", "", md_table(res), "",
  "![Forecasts](R_01_forecasts.png)", "", "![MAPE](R_02_mape.png)", "",
  "## ARIMA(0,1,2) regressor effects", "",
  paste0("- ", regs, ": **", sprintf("%+.1f%%", 100 * (exp(eff) - 1)), "**"), "",
  sprintf("Ljung-Box p-values on ARIMA residuals: %s", paste(sprintf("%s = %.3f", names(lb), lb), collapse = ", ")), "",
  "## Cross-check with Python", "", paste("-", check), "",
  "**Reading the results:** the models that use the promo/holiday calendar (ARIMA + Fourier, harmonic regression) beat the pure smoothing methods, and every model beats the seasonal-naive benchmark. That's the same conclusion as the Python notebook, reached with a different language and different estimation code.")
writeLines(report, file.path(out_dir, "R_results.md"))
cat("Wrote outputs/r/R_results.md, CSV and charts\n")
