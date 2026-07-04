TEX_DIR := tex
TEX_MAIN := linked_beverton_holt_fishing.tex
PDFLATEX := pdflatex
PYTHON := ./.venv/Scripts/python.exe -B
COBWEB_PLOT := $(TEX_DIR)/plots/beverton_holt_cobweb_r1_4_k1.pdf
FISHING_PLOT := $(TEX_DIR)/plots/fishing_equilibrium_r1_4_k1.pdf
YIELD_PLOT := $(TEX_DIR)/plots/fishing_yield_r1_4_k1.pdf
LINKED_NO_FISHING_PLOT := $(TEX_DIR)/plots/linked_no_fishing_extinction_m0_2.pdf
LINKED_CONNECTION_PLOT := $(TEX_DIR)/plots/linked_connection_effect_ra1_8_rb1_2.pdf
POPULATION_DOMAIN_YIELD_VIEW_PLOTS := \
	$(TEX_DIR)/plots/population_domain_view_interior.pdf \
	$(TEX_DIR)/plots/population_domain_view_complete_harvest.pdf \
	$(TEX_DIR)/plots/population_domain_view_source_endpoint.pdf \
	$(TEX_DIR)/plots/population_domain_view_no_take.pdf
NUMERIC_YIELD_ZONE_MAP := $(TEX_DIR)/plots/numeric_yield_zone_map_ka2_kb1_m0_45.pdf
LINKED_EXTINCTION_PLOTS := \
	$(TEX_DIR)/plots/linked_extinction_map_ra1_4_rb0_8_m0_3.pdf \
	$(TEX_DIR)/plots/linked_extinction_map_ra1_4_rb1_2_m0_1.pdf \
	$(TEX_DIR)/plots/linked_extinction_map_ra1_4_rb1_2_m0_3.pdf \
	$(TEX_DIR)/plots/linked_extinction_map_ra1_4_rb1_2_m0_2.pdf \
	$(TEX_DIR)/plots/linked_extinction_map_ra1_4_rb1_2_m1_0.pdf \
	$(TEX_DIR)/plots/linked_extinction_map_ra2_0_rb1_2_m0_1.pdf
PLOTS := $(COBWEB_PLOT) $(FISHING_PLOT) $(YIELD_PLOT) $(LINKED_NO_FISHING_PLOT) $(LINKED_CONNECTION_PLOT) $(POPULATION_DOMAIN_YIELD_VIEW_PLOTS) $(NUMERIC_YIELD_ZONE_MAP) $(LINKED_EXTINCTION_PLOTS)
PLOT_COMMON := plot_generators/common.py

.PHONY: all pdf clean

all: pdf

pdf: $(PLOTS)
	cd $(TEX_DIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(TEX_MAIN)
	cd $(TEX_DIR) && $(PDFLATEX) -interaction=nonstopmode -halt-on-error $(TEX_MAIN)

$(COBWEB_PLOT): plot_generators/beverton_holt_cobweb.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.beverton_holt_cobweb

$(FISHING_PLOT): plot_generators/fishing_equilibrium.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.fishing_equilibrium

$(YIELD_PLOT): plot_generators/fishing_yield.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.fishing_yield

$(LINKED_NO_FISHING_PLOT): plot_generators/linked_no_fishing_extinction.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.linked_no_fishing_extinction

$(LINKED_CONNECTION_PLOT): plot_generators/linked_connection_effect.py linked_model.py beverton_holt.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.linked_connection_effect

$(POPULATION_DOMAIN_YIELD_VIEW_PLOTS) &: plot_generators/population_domain_yield_views.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.population_domain_yield_views

$(NUMERIC_YIELD_ZONE_MAP): plot_generators/numeric_yield_zone_map.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.numeric_yield_zone_map --m 0.45 --k-a 2 --k-b 1 --grid-size 151 --effort-grid-size 40 --local-effort-grid-size 100 --steps 30 --local-steps 300 --output $(NUMERIC_YIELD_ZONE_MAP)
	powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue '$(NUMERIC_YIELD_ZONE_MAP:.pdf=.png)'"

$(LINKED_EXTINCTION_PLOTS) &: plot_generators/linked_extinction_maps.py $(PLOT_COMMON)
	$(PYTHON) -m plot_generators.linked_extinction_maps

clean:
	cd $(TEX_DIR) && powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue *.aux,*.log,*.out,*.toc,*.fls,*.fdb_latexmk,*.synctex.gz,*.pdf"
	powershell -NoProfile -Command "Remove-Item -Force -ErrorAction SilentlyContinue '$(TEX_DIR)/plots/*.pdf'"
