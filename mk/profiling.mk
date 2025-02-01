PROFILING_DIR = $(STORAGE_DIR)/output/profiling
PROFILING_TARGET = profiling.nn

profile:
	$(POETRY_RUN) python -um profiling.profile -t $(PROFILING_TARGET)

profile-once:
	[ -f $(PROFILING_DIR)/$(PROFILING_TARGET).stats ] || $(MAKE) profile

profile-png: profile-once profile-create-png

profile-create-png:
	gprof2dot -f pstats $(PROFILING_DIR)/$(PROFILING_TARGET).stats \
--node-label=total-time \
--node-label=total-time-percentage \
--node-label=self-time-percentage \
| dot -Tpng -o $(PROFILING_DIR)/$(PROFILING_TARGET).stats.png \
&& xdg-open $(PROFILING_DIR)/$(PROFILING_TARGET).stats.png
