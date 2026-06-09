# ================================================================
# UNIFIED LOCAL ANCESTRY INFERENCE SNAKEFILE
# Methods: AncestryHMM | GNOMix | RFMix
# ================================================================
#
# Dry run:
# snakemake -n -p -s /shared/projects/admixture_yeast/LocAncInf/scripts/0_LAIpipeline/LAIpipeline.snakefile --rerun-triggers mtime --rerun-incomplete
#
# Launch:
# nohup snakemake --snakefile /shared/projects/admixture_yeast/LocAncInf/scripts/0_LAIpipeline/LAIpipeline.snakefile --jobs 200 --max-jobs-per-second 5 --max-status-checks-per-second 2 --rerun-triggers mtime --rerun-incomplete --cluster "sbatch -p {resources.partition} -t {resources.time} --mem={resources.mem_mb}M --cpus-per-task={resources.cpus_per_task} -J LAI_{rule} -o /shared/projects/admixture_yeast/LocAncInf/scripts/0_LAIpipeline/logs/slurm_{rule}_{jobid}.out -e /shared/projects/admixture_yeast/LocAncInf/scripts/0_LAIpipeline/logs/slurm_{rule}_{jobid}.err" -p &> /shared/projects/admixture_yeast/LocAncInf/scripts/0_LAIpipeline/logs/nohup_LAI.out &
# ================================================================

import pandas as pd
import re
import os

# ================================================================
# CONFIGURATION — PATHS TO FILES
# ================================================================

input_VCF              = "/shared/projects/admixture_yeast/VCFs/filtered_repremoved_yeast_fixed_phased_total_MAC1.vcf.gz"
list_ID_parentalpop    = "/shared/projects/admixture_yeast/Populations/sampleID_Pop.tsv"
list_ID_clusteradmixed = "/shared/projects/admixture_yeast/Populations/sampleID_clusteradmixed.tsv"
list_cluadmixed_parpop = "/shared/projects/admixture_yeast/Populations/ClusterAdmID_top2.tsv"
pop_colours            = "/shared/projects/admixture_yeast/Populations/pop_colours.tsv"
sampleID_ploidy        = "/shared/projects/admixture_yeast/Populations/SampleID_Ploidy.tsv"
morgan_distances       = "/shared/projects/admixture_yeast/LocAncInf/input_tsvs/sites_with_distances.tsv"
sample_proportions     = "/shared/projects/admixture_yeast/Populations/SampleID_proportions.tsv"

# ================================================================
# CONFIGURATION — PATHS TO DIRECTORIES
# ================================================================

VCFs_dir         = "/shared/projects/admixture_yeast/LocAncInf/input_VCFs"
tsv_dir          = "/shared/projects/admixture_yeast/LocAncInf/input_tsvs"
DFs_dir          = "/shared/projects/admixture_yeast/LocAncInf/input_DF"
recombmap_dir    = "/shared/projects/admixture_yeast/RecombMaps/separateCHR"
GNOMix_tool_dir  = "/shared/projects/admixture_yeast/tools/gnomix"
ancestryHMM_tool  = "/shared/projects/admixture_yeast/tools/ancestryHMM/Ancestry_HMM/src/ancestry_hmm"

AncestryHMM_out_dir = "/shared/projects/admixture_yeast/LocAncInf/OUTPUTS/Ancestry_HMM"
GNOMix_out_dir      = "/shared/projects/admixture_yeast/LocAncInf/OUTPUTS/GNOMix"
rfmix_out_dir       = "/shared/projects/admixture_yeast/LocAncInf/OUTPUTS/RFMix"
consensus_dir       = "/shared/projects/admixture_yeast/LocAncInf/1_CONSENSUS"
global_anc_dir      = "/shared/projects/admixture_yeast/LocAncInf/2_METRICS/GlobalAnc"
excess_anc_dir      = "/shared/projects/admixture_yeast/LocAncInf/2_METRICS/ExcessAnc"
tract_length_dir    = "/shared/projects/admixture_yeast/LocAncInf/2_METRICS/TractLength"

# Scripts
script_hmm_contrib  = "/shared/projects/admixture_yeast/LocAncInf/scripts/AncestryHMM/create_cluster_contributions.py"
script_consensus    = "/shared/projects/admixture_yeast/LocAncInf/scripts/1_Consensus/Consensus3methods.py"
script_phasing      = "/shared/projects/admixture_yeast/LocAncInf/scripts/1_Consensus/ConsensusPhasing.py"
script_global_anc   = "/shared/projects/admixture_yeast/LocAncInf/scripts/2_Metrics/global_ancestry.py"
script_anc_comp     = "/shared/projects/admixture_yeast/LocAncInf/scripts/2_Metrics/global_ancestry_comparison.py"
script_excess_binom = "/shared/projects/admixture_yeast/LocAncInf/scripts/2_Metrics/excess_ancestry_binom.py"
script_excess_dev   = "/shared/projects/admixture_yeast/LocAncInf/scripts/2_Metrics/excess_ancestry_deviation.py"
script_tract_length = "/shared/projects/admixture_yeast/LocAncInf/scripts/2_Metrics/tractlength_distribution.py"

# Chromosomes
CHROMS = list(range(1, 17))

# ================================================================
# CLUSTER CONFIGURATION
# ================================================================

def load_cluster_info(tsv_file):
    """Load all cluster → (p1, p2) mappings from the TSV."""
    df = pd.read_csv(tsv_file, sep='\t')
    df.columns = df.columns.str.strip()
    for col in df.columns:
        df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    info = {}
    for _, row in df.iterrows():
        cid = str(int(row['admixed_cluster']))
        info[cid] = {
            'p1': re.sub(r'\s+', '', str(row['source1'])),
            'p2': re.sub(r'\s+', '', str(row['source2']))
        }
    return info

def load_admixed_samples(tsv_file):
    """Load cluster → sample list from the admixed samples TSV (available at DAG build time)."""
    df = pd.read_csv(tsv_file, sep='\t', header=0)
    df.columns = df.columns.str.strip()
    col_sample  = df.columns[0]
    col_cluster = df.columns[1]
    samples_by_cluster = {}
    for _, row in df.iterrows():
        cid = str(row[col_cluster]).strip()
        sid = str(row[col_sample]).strip()
        samples_by_cluster.setdefault(cid, []).append(sid)
    return samples_by_cluster

CLUSTER_INFO    = load_cluster_info(list_cluadmixed_parpop)
CLUSTER_SAMPLES = load_admixed_samples(list_ID_clusteradmixed)

# ▶▶ EDIT THIS LIST to choose which clusters to run (or use list(CLUSTER_INFO.keys()) for all)
CLUSTERS = [
    "2", "4", "9", "12", "17", "18", "25", "33", "35", "37",
    "40", "41", "42", "43", "44", "52", "53", "55", "58", "62",
    "63", "67", "70", "73", "75", "76", "80", "85", "90", "119"]

# Clusters avec ≥ 2 individus admixés (requis pour global/excess ancestry)
CLUSTERS_MIN2 = [c for c in CLUSTERS if len(CLUSTER_SAMPLES.get(c, [])) >= 2]

# Helper functions — p1/p2 resolved from wildcard at runtime
def get_p1(wildcards):
    return CLUSTER_INFO[str(wildcards.cluster)]['p1']

def get_p2(wildcards):
    return CLUSTER_INFO[str(wildcards.cluster)]['p2']

# Print summary at workflow start
print("=" * 55)
print("  UNIFIED LAI WORKFLOW")
print("=" * 55)
for c in CLUSTERS:
    print("  Cluster {:>4s}  |  p1 = {}  |  p2 = {}".format(c, CLUSTER_INFO[c]['p1'], CLUSTER_INFO[c]['p2']))
print("=" * 55)
skipped = set(CLUSTERS) - set(CLUSTERS_MIN2)
if skipped:
    print("  [INFO] Clusters exclus (< 2 individus) pour Excess Ancestry :", ", ".join(sorted(skipped, key=int)))
print("=" * 55)


# ================================================================
# RULE ALL
# ================================================================

rule all:
    input:
        # ── AncestryHMM ───────────────────────────────────────
        expand(
            AncestryHMM_out_dir + "/cluster_{cluster}/chr{chrom}/ancestryHMM_chr{chrom}.done",
            cluster=CLUSTERS, chrom=CHROMS),

        # ── GNOMix ────────────────────────────────────────────
        expand(
            GNOMix_out_dir + "/cluster_{cluster}/cluster_{cluster}_chr{chrom}.msp",
            cluster=CLUSTERS, chrom=CHROMS),

        # ── RFMix ─────────────────────────────────────────────
        expand(
            rfmix_out_dir + "/cluster_{cluster}/cluster_{cluster}_chr{chrom}.msp.tsv",
            cluster=CLUSTERS, chrom=CHROMS),

        # ── Consensus ─────────────────────────────────────────
        expand(
            consensus_dir + "/cluster_{cluster}/cluster_{cluster}_genomewide_detailed_comparison.tsv",
            cluster=CLUSTERS),

        # ── Phasing ───────────────────────────────────────────
        expand(
            consensus_dir + "/cluster_{cluster}/cluster_{cluster}_genomewide_detailed_comparison_phased.tsv",
            cluster=CLUSTERS),

        # ── Global Ancestry (≥ 2 individus seulement) ─────────
        expand(
            global_anc_dir + "/cluster_{cluster}/ancestry_proportions.tsv",
            cluster=CLUSTERS),
        ([global_anc_dir + "/comparaison.svg"] if CLUSTERS else []),

        # ── Excess ancestry (≥ 2 individus seulement) ─────────
        expand(
            excess_anc_dir + "/cluster_{cluster}/excess_windows_binom.tsv",
            cluster=CLUSTERS_MIN2),
        expand(
            excess_anc_dir + "/cluster_{cluster}/deviation_windows.tsv",
            cluster=CLUSTERS_MIN2),

        # ── Tract length ──────────────────────────────────────
        expand(
            tract_length_dir + "/cluster_{cluster}/tract_length_distribution.svg",
            cluster=CLUSTERS)

# ================================================================
# SHARED PREPROCESSING RULES  (run once per cluster → used by all 3 methods)
# ================================================================

rule get_sampleIDs:
    """Extract sample IDs for admixed cluster and both parental populations."""
    input:
        tsv_admixed  = ancient(list_ID_clusteradmixed),
        tsv_parental = ancient(list_ID_parentalpop)
    output:
        admixed = tsv_dir + "/cluster_{cluster}/admixed_samples.txt",
        pop1    = tsv_dir + "/cluster_{cluster}/pop1_samples.txt",
        pop2    = tsv_dir + "/cluster_{cluster}/pop2_samples.txt"
    params:
        p1 = get_p1,
        p2 = get_p2
    resources:
        mem_mb=200, time="00:05:00", cpus_per_task=1, partition="fast"
    shell:
        """
        mkdir -p {tsv_dir}/cluster_{wildcards.cluster}

        awk -F'\t' 'NR>1 && $2=="{wildcards.cluster}" {{print $1}}' {input.tsv_admixed} > {output.admixed}
        awk -F'\t' 'NR>1 && $2=="{params.p1}" {{print $1}}' {input.tsv_parental}         > {output.pop1}
        awk -F'\t' 'NR>1 && $2=="{params.p2}" {{print $1}}' {input.tsv_parental}         > {output.pop2}

        echo "[cluster {wildcards.cluster}] Admixed samples : $(wc -l < {output.admixed})"
        echo "[cluster {wildcards.cluster}] Pop1 ({params.p1}) : $(wc -l < {output.pop1})"
        echo "[cluster {wildcards.cluster}] Pop2 ({params.p2}) : $(wc -l < {output.pop2})"
        """


rule extract_trio_vcf:
    """Subset main VCF to the trio (admixed + both parentals). Temporary file."""
    input:
        vcf     = input_VCF,
        admixed = tsv_dir + "/cluster_{cluster}/admixed_samples.txt",
        pop1    = tsv_dir + "/cluster_{cluster}/pop1_samples.txt",
        pop2    = tsv_dir + "/cluster_{cluster}/pop2_samples.txt"
    output:
        temp(VCFs_dir + "/cluster_{cluster}/trio.vcf")
    resources:
        mem_mb=700, time="00:10:00", cpus_per_task=1, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_env/vcf_env/bin:$PATH
        
        mkdir -p {VCFs_dir}/cluster_{wildcards.cluster}

        TMPSAMPLES={VCFs_dir}/cluster_{wildcards.cluster}/trio_samples.tmp
        cat {input.admixed} {input.pop1} {input.pop2} > $TMPSAMPLES
        echo "Total trio samples: $(wc -l < $TMPSAMPLES)"

        bcftools view -S $TMPSAMPLES {input.vcf} -o {output}
        rm -f $TMPSAMPLES

        echo "Variants in trio VCF (before filtering): $(bcftools view -H {output} | wc -l)"
        if [ ! -f {output} ]; then echo "ERROR: Output VCF not created"; exit 1; fi
        sync
        """


rule filter_trio_variants:
    """Keep only variant sites (AC > 0) in the trio VCF."""
    input:
        VCFs_dir + "/cluster_{cluster}/trio.vcf"
    output:
        VCFs_dir + "/cluster_{cluster}/trio_variantsonly.vcf"
    resources:
        mem_mb=500, time="00:05:00", cpus_per_task=1, partition="fast"
    shell:
        """
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_env/vcf_env/bin:$PATH

        bcftools view -i 'AC>0' {input} -o {output}

        echo "Variants kept: $(bcftools view -H {output} | wc -l)"
        n_mono=$(bcftools query -f '%AC\n' {output} | grep -c "^0$" || echo "0")
        echo "Monomorphic sites remaining (should be 0): $n_mono"
        n_sing=$(bcftools query -f '%AC\n' {output} | grep -c "^1$" || echo "0")
        echo "Singletons (AC=1): $n_sing"

        if [ ! -f {output} ]; then echo "ERROR: Output VCF not created"; exit 1; fi
        sync
        """


rule extract_admixed_vcf:
    """Extract admixed samples from the filtered trio VCF."""
    input:
        vcf     = VCFs_dir + "/cluster_{cluster}/trio_variantsonly.vcf",
        samples = tsv_dir  + "/cluster_{cluster}/admixed_samples.txt"
    output:
        VCFs_dir + "/admixed/cluster_{cluster}.vcf"
    resources:
        mem_mb=300, time="00:02:00", cpus_per_task=1, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_env/vcf_env/bin:$PATH

        mkdir -p {VCFs_dir}/admixed
        bcftools view -S {input.samples} {input.vcf} -o {output}

        if [ ! -f {output} ]; then echo "ERROR: Output VCF not created"; exit 1; fi
        echo "Variants in admixed VCF: $(bcftools view -H {output} | wc -l)"
        sync
        """


rule create_sampleID_pop_file:
    """Create sample→population mapping file (shared by GNOMix and RFMix)."""
    input:
        pop1_samples = tsv_dir + "/cluster_{cluster}/pop1_samples.txt",
        pop2_samples = tsv_dir + "/cluster_{cluster}/pop2_samples.txt",
        mapping_file = ancient(list_ID_parentalpop)
    output:
        tsv_dir + "/parental/cluster_{cluster}_sampleID_pop.tsv"
    params:
        p1 = get_p1,
        p2 = get_p2
    resources:
        mem_mb=200, time="00:02:00", cpus_per_task=1, partition="fast"
    shell:
        """
        mkdir -p {tsv_dir}/parental
        {{
            while IFS=$'\t' read -r sample pop; do
                pop_clean=$(echo "$pop" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
                if [ "$pop_clean" = "{params.p1}" ] || [ "$pop_clean" = "{params.p2}" ]; then
                    echo -e "$sample\t$pop_clean"
                fi
            done < <(tail -n +2 {input.mapping_file})
        }} > {output}
        echo "Samples in mapping file: $(wc -l < {output})"
        """


# ================================================================
# ANCESTRYHMM — SPECIFIC RULES
# ================================================================

rule extract_p1_vcf:
    """Extract parental pop1 VCF for AncestryHMM allele count dataframe."""
    input:
        vcf     = VCFs_dir + "/cluster_{cluster}/trio_variantsonly.vcf",
        samples = tsv_dir  + "/cluster_{cluster}/pop1_samples.txt"
    output:
        VCFs_dir + "/parental/cluster_{cluster}_pop1.vcf"
    resources:
        mem_mb=1000, time="00:02:00", cpus_per_task=1, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_env/vcf_env/bin:$PATH

        mkdir -p {VCFs_dir}/parental
        bcftools view -S {input.samples} {input.vcf} -o {output}

        if [ ! -f {output} ]; then echo "ERROR: Output VCF not created"; exit 1; fi
        sync
        """


rule extract_p2_vcf:
    """Extract parental pop2 VCF for AncestryHMM allele count dataframe."""
    input:
        vcf     = VCFs_dir + "/cluster_{cluster}/trio_variantsonly.vcf",
        samples = tsv_dir  + "/cluster_{cluster}/pop2_samples.txt"
    output:
        VCFs_dir + "/parental/cluster_{cluster}_pop2.vcf"
    resources:
        mem_mb=1000, time="00:02:00", cpus_per_task=1, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_env/vcf_env/bin:$PATH

        mkdir -p {VCFs_dir}/parental
        bcftools view -S {input.samples} {input.vcf} -o {output}

        if [ ! -f {output} ]; then echo "ERROR: Output VCF not created"; exit 1; fi
        sync
        """


rule create_admixed_dataframe:
    """Build per-sample allele-count dataframe from the admixed VCF."""
    input:
        vcf = VCFs_dir + "/admixed/cluster_{cluster}.vcf"
    output:
        DFs_dir + "/admixed/AncHMM_df_admixed_cluster_{cluster}.tsv"
    resources:
        mem_mb=600, time="00:10:00", cpus_per_task=1, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_env/vcf_env/bin:$PATH

        mkdir -p {DFs_dir}/admixed

        samples=$(bcftools query -l {input.vcf})
        header="#CHROM\tPOS"
        for sample in $samples; do
            header="${{header}}\t${{sample}}_0\t${{sample}}_1"
        done

        (
          echo -e "$header"
          bcftools query -f '%CHROM\t%POS[\t%GT]\n' {input.vcf} \
          | awk 'BEGIN{{OFS="\t"}}{{
              printf "%s\t%s", $1, $2
              for (i=3;i<=NF;i++){{
                n0=gsub(/0/,"",$i); n1=gsub(/1/,"",$i)
                printf "\t%d\t%d", n0, n1
              }}
              print ""
          }}'
        ) > {output}
        """


rule create_parental_dataframe:
    """Aggregate allele counts across all individuals in each parental population."""
    input:
        vcf_p1 = VCFs_dir + "/parental/cluster_{cluster}_pop1.vcf",
        vcf_p2 = VCFs_dir + "/parental/cluster_{cluster}_pop2.vcf"
    output:
        DFs_dir + "/parental/AncHMM_df_parental_combined_cluster_{cluster}.tsv"
    resources:
        mem_mb=600, time="00:20:00", cpus_per_task=1, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_env/vcf_env/bin:$PATH

        mkdir -p {DFs_dir}/parental

        TMP1={DFs_dir}/parental/tmp_p1_cluster_{wildcards.cluster}.tsv
        TMP2={DFs_dir}/parental/tmp_p2_cluster_{wildcards.cluster}.tsv

        bcftools query -f '%CHROM\t%POS[\t%GT]\n' {input.vcf_p1} \
        | awk 'BEGIN{{OFS="\t"}}{{
            t0=0; t1=0
            for(i=3;i<=NF;i++){{ t0+=gsub(/0/,"",$i); t1+=gsub(/1/,"",$i) }}
            print $1,$2,t0,t1
        }}' > $TMP1

        bcftools query -f '%CHROM\t%POS[\t%GT]\n' {input.vcf_p2} \
        | awk 'BEGIN{{OFS="\t"}}{{
            t0=0; t1=0
            for(i=3;i<=NF;i++){{ t0+=gsub(/0/,"",$i); t1+=gsub(/1/,"",$i) }}
            print $1,$2,t0,t1
        }}' > $TMP2

        paste $TMP1 $TMP2 \
        | awk 'BEGIN{{OFS="\t"}}{{print $1,$2,$3,$4,$7,$8}}' \
        | cat <(echo -e "#CHROM\tPOS\tpop0_allele0\tpop0_allele1\tpop1_allele0\tpop1_allele1") - \
        > {output}

        rm -f $TMP1 $TMP2
        """


rule combine_all_dataframes:
    """Inner-join parental counts, Morgan distances, and admixed allele counts."""
    input:
        parental  = DFs_dir + "/parental/AncHMM_df_parental_combined_cluster_{cluster}.tsv",
        distances = morgan_distances,
        admixed   = DFs_dir + "/admixed/AncHMM_df_admixed_cluster_{cluster}.tsv"
    output:
        DFs_dir + "/DFall_cluster_{cluster}.tsv"
    resources:
        mem_mb=3000, time="00:10:00", cpus_per_task=1, partition="fast"
    shell:
        """
        awk -F"\t" 'BEGIN {{OFS="\t"}}
        ARGIND==1 {{
            if (FNR==1) next
            key=$1"-"$2; parental[key]=$3"\t"$4"\t"$5"\t"$6; next
        }}
        ARGIND==2 {{
            if (FNR==1) next
            key=$1"-"$2; distances[key]=$4; next
        }}
        ARGIND==3 {{
            if (FNR==1) {{
                printf "CHROM\tPOS\tpop0_allele0\tpop0_allele1\tpop1_allele0\tpop1_allele1\tDIST_MORGAN"
                for (i=3;i<=NF;i++) printf "\t%s",$i
                print ""; next
            }}
            key=$1"-"$2
            if (key in parental && key in distances) {{
                printf "%s\t%s\t%s\t%s",$1,$2,parental[key],distances[key]
                for (i=3;i<=NF;i++) printf "\t%s",$i
                print ""
            }}
        }}' {input.parental} {input.distances} {input.admixed} > {output}

        echo "=== DFall cluster {wildcards.cluster} ==="
        echo "Total lines: $(wc -l < {output})"
        head -3 {output}
        """


rule get_admixed_ploidy:
    """Extract ploidy for each admixed sample."""
    input:
        samples = tsv_dir + "/cluster_{cluster}/admixed_samples.txt",
        ploidy  = ancient(sampleID_ploidy)
    output:
        ploidy_file = tsv_dir + "/cluster_{cluster}/admixed_ploidy.tsv",
        log_file    = tsv_dir + "/cluster_{cluster}/admixed_ploidy.out"
    resources:
        mem_mb=300, time="00:05:00", cpus_per_task=1, partition="fast"
    shell:
        """
        echo "=== Ploidy extraction log ===" > {output.log_file}
        echo "Date: $(date)" >> {output.log_file}
        echo "Cluster: {wildcards.cluster}" >> {output.log_file}

        awk -F'\t' '
            NR==FNR {{samples[$1]; next}}
            FNR>1 && $1 in samples {{
                if ($2=="NA" || $2=="") {{
                    print $1"\tNA"
                    print "WARNING "$1": ploidy is NA" > "/dev/stderr"
                }} else {{
                    print $1"\t"$2
                }}
            }}
        ' {input.samples} {input.ploidy} 2>>{output.log_file} > {output.ploidy_file}

        echo "" >> {output.log_file}
        echo "Total samples: $(wc -l < {output.ploidy_file})" >> {output.log_file}
        echo "Ploidy distribution:" >> {output.log_file}
        cut -f2 {output.ploidy_file} | sort | uniq -c \
            | awk '{{printf "  Ploidy %s: %d samples\\n", $2, $1}}' >> {output.log_file}

        echo "Ploidy file created for cluster {wildcards.cluster}"
        cat {output.log_file}
        """


rule create_cluster_contributions:
    """Compute admixture proportions (pop0/pop1) per sample for AncestryHMM."""
    input:
        proportions = ancient(sample_proportions),
        samples     = tsv_dir + "/cluster_{cluster}/admixed_samples.txt"
    output:
        tsv_dir + "/cluster_{cluster}/admixed_contributions.tsv"
    params:
        p1     = get_p1,
        p2     = get_p2,
        script = script_hmm_contrib
    resources:
        mem_mb=500, time="00:10:00", cpus_per_task=1, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_envs/plot_env/bin:$PATH

        /shared/projects/admixture_yeast/LocAncInf/conda_envs/plot_env/bin/python {params.script} \
          --proportions {input.proportions} \
          --samples     {input.samples} \
          --p1          "{params.p1}" \
          --p2          "{params.p2}" \
          --output      {output}
        """


rule subset_df_per_sample_chr:
    """Slice the combined dataframe to one sample × one chromosome."""
    input:
        df = DFs_dir + "/DFall_cluster_{cluster}.tsv"
    output:
        temp(DFs_dir + "/tmp/cluster_{cluster}/{sample}_chr{chrom}.tsv")
    params:
        sample = "{sample}"
    resources:
        mem_mb=400, time="00:05:00", cpus_per_task=1, partition="fast"
    shell:
        """
        mkdir -p {DFs_dir}/tmp/cluster_{wildcards.cluster}

        header=$(head -1 "{input.df}")
        col_indices=$(echo "$header" | awk -F'\t' -v sample="{params.sample}" '{{
            for(i=1;i<=NF;i++) {{
                if ($i==sample"_0" || $i==sample"_1") {{
                    cols = cols ? cols","i : i
                }}
            }}
            print cols
        }}')

        awk -F'\t' -v chrom="{wildcards.chrom}" -v cols="$col_indices" '
        BEGIN {{ OFS="\t"; split(cols, col_arr, ",") }}
        NR>1 && $1==chrom {{
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s", $1,$2,$3,$4,$5,$6,$7
            for (i in col_arr) printf "\t%s", $col_arr[i]
            print ""
        }}' "{input.df}" > "{output}"

        echo "Subset {params.sample} chr{wildcards.chrom}: $(wc -l < "{output}") lines"
        if [ ! -f "{output}" ]; then echo "ERROR: Output file not created"; exit 1; fi
        sync
        """


rule subset_ploidy_per_sample:
    """Force ploidy=2 for all samples (required by Ancestry_HMM input format)."""
    input:
        samples = tsv_dir + "/cluster_{cluster}/admixed_samples.txt"
    output:
        temp(tsv_dir + "/tmp/cluster_{cluster}/{sample}_ploidy.tsv")
    params:
        sample = "{sample}"
    resources:
        mem_mb=300, time="00:02:00", cpus_per_task=1, partition="fast"
    shell:
        """
        mkdir -p {tsv_dir}/tmp/cluster_{wildcards.cluster}

        grep '^{params.sample}$' "{input.samples}" | awk 'BEGIN {{OFS="\t"}} {{print $1, 2}}' > "{output}"

        if [ ! -s "{output}" ]; then
            echo "ERROR: Sample '{params.sample}' not found in samples file!" >&2
            exit 1
        fi
        """


rule get_sample_contributions:
    """Extract contribution row (pop0/pop1 proportions) for a single sample."""
    input:
        contrib = tsv_dir + "/cluster_{cluster}/admixed_contributions.tsv"
    output:
        temp(tsv_dir + "/tmp/cluster_{cluster}/{sample}_contrib.txt")
    params:
        sample = "{sample}"
    resources:
        mem_mb=300, time="00:03:00", cpus_per_task=1, partition="fast"
    shell:
        """
        mkdir -p {tsv_dir}/tmp/cluster_{wildcards.cluster}

        awk -F'\t' 'NR>1 && $1=="{params.sample}" {{print $2, $3}}' "{input.contrib}" > "{output}"

        if [ ! -s "{output}" ]; then
            echo "ERROR: Sample '{params.sample}' not found in contributions file!" >&2
            exit 1
        fi
        """


def get_viterbi_files(wildcards):
    """Collect all per-sample viterbi paths for a given cluster × chromosome.
    Uses CLUSTER_SAMPLES (loaded from source TSV at startup) to avoid
    reading a file that does not exist yet at DAG-building time.
    """
    samples = CLUSTER_SAMPLES.get(str(wildcards.cluster), [])
    if not samples:
        raise ValueError(f"No samples found for cluster {wildcards.cluster} in CLUSTER_SAMPLES")
    return expand(
        AncestryHMM_out_dir + "/cluster_{cluster}/chr{chrom}/{sample}.viterbi",
        cluster=wildcards.cluster,
        chrom=wildcards.chrom,
        sample=samples
    )


rule ancestry_hmm_per_sample_chr:
    """Run Ancestry_HMM for one sample on one chromosome."""
    input:
        df      = DFs_dir + "/tmp/cluster_{cluster}/{sample}_chr{chrom}.tsv",
        ploidy  = tsv_dir + "/tmp/cluster_{cluster}/{sample}_ploidy.tsv",
        contrib = tsv_dir + "/tmp/cluster_{cluster}/{sample}_contrib.txt"
    output:
        viterbi = AncestryHMM_out_dir + "/cluster_{cluster}/chr{chrom}/{sample}.viterbi"
    log:
        out = AncestryHMM_out_dir + "/cluster_{cluster}/chr{chrom}/logs/{sample}.out",
        err = AncestryHMM_out_dir + "/cluster_{cluster}/chr{chrom}/logs/{sample}.err"
    params:
        outdir = AncestryHMM_out_dir + "/cluster_{cluster}/chr{chrom}"
    resources:
        mem_mb=5000, time="00:30:00", cpus_per_task=2, partition="fast"
    shell:
        """
        set +euo pipefail

        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_envs/ancestryHMM_env/bin:$PATH
        export LD_LIBRARY_PATH=/shared/projects/admixture_yeast/LocAncInf/conda_envs/ancestryHMM_env/lib:$LD_LIBRARY_PATH
        
        mkdir -p "{params.outdir}/logs"
        cd "{params.outdir}"

        read pop0_contrib pop1_contrib < "{input.contrib}"

        echo "Running Ancestry_HMM — cluster {wildcards.cluster} / {wildcards.sample} / chr{wildcards.chrom}" > "{log.out}"
        echo "Contributions: pop0=$pop0_contrib  pop1=$pop1_contrib" >> "{log.out}"
        echo "Input lines: $(wc -l < "{input.df}")" >> "{log.out}"

        {ancestryHMM_tool} \
          -i "{input.df}" \
          -s "{input.ploidy}" \
          -a 2 $pop0_contrib $pop1_contrib \
          -p 0 1000000 $pop0_contrib \
          -p 1 -3000   $pop1_contrib \
          -v -g \
          --tmax 20000 \
          >> "{log.out}" 2> "{log.err}"

        # ancestry_hmm writes output to cwd using sample name from ploidy file
        SAMPLE_VITERBI="{wildcards.sample}.viterbi"
        if [ -f "$SAMPLE_VITERBI" ]; then
            echo "Success: viterbi file created" >> "{log.out}"
        else
            echo "ERROR: .viterbi file not created!" >> "{log.err}"
            ls -la >> "{log.err}"
            exit 1
        fi

        if [ ! -f "{output.viterbi}" ]; then
            echo "ERROR: expected output not at {output.viterbi}" >> "{log.err}"
            exit 1
        fi
        sync
        """


rule ancestryHMM_done_per_chrom:
    """Sentinel rule: all samples done for this cluster × chromosome."""
    input:
        viterbi_files = get_viterbi_files
    output:
        AncestryHMM_out_dir + "/cluster_{cluster}/chr{chrom}/ancestryHMM_chr{chrom}.done"
    resources:
        mem_mb=200, time="00:02:00", cpus_per_task=1, partition="fast"
    shell:
        """
        echo "All viterbi files complete — cluster {wildcards.cluster} chr{wildcards.chrom}" > {output}
        """


# ================================================================
# GNOMIX — SPECIFIC RULES
# ================================================================

rule run_GNOMix:
    """Run GNOMix for one cluster × chromosome."""
    input:
        adm_vcf       = VCFs_dir + "/admixed/cluster_{cluster}.vcf",
        ref_vcf       = VCFs_dir + "/cluster_{cluster}/trio_variantsonly.vcf",
        refpop_labels = tsv_dir  + "/parental/cluster_{cluster}_sampleID_pop.tsv",
        recomb_map    = recombmap_dir + "/map_chr{chrom}"
    output:
        msp = GNOMix_out_dir + "/cluster_{cluster}/cluster_{cluster}_chr{chrom}.msp"
    params:
        outdir   = GNOMix_out_dir + "/cluster_{cluster}",
        temp_out = GNOMix_out_dir + "/cluster_{cluster}/temp_chr{chrom}"
    resources:
        mem_mb=90000, time="12:00:00", cpus_per_task=10, partition="fast"
    shell:
        """
        mkdir -p {params.outdir} {params.temp_out}
        cd {GNOMix_tool_dir}

        /shared/projects/admixture_yeast/LocAncInf/conda_envs/GNOMix_env/bin/python {GNOMix_tool_dir}/gnomix.py \
          {input.adm_vcf} \
          {params.temp_out} \
          {wildcards.chrom} \
          True \
          {input.recomb_map} \
          {input.ref_vcf} \
          {input.refpop_labels}

        mv {params.temp_out}/query_results.msp {output.msp}
        rm -rf {params.temp_out}

        if [ ! -f {output.msp} ]; then echo "ERROR: GNOMix .msp not created"; exit 1; fi
        sync
        """


# ================================================================
# RFMIX — SPECIFIC RULES
# ================================================================

rule run_RFMix:
    """Run RFMix for one cluster × chromosome."""
    input:
        adm_vcf    = VCFs_dir + "/admixed/cluster_{cluster}.vcf",
        ref_vcf    = VCFs_dir + "/cluster_{cluster}/trio_variantsonly.vcf",
        sample_map = tsv_dir  + "/parental/cluster_{cluster}_sampleID_pop.tsv"
    output:
        msp = rfmix_out_dir + "/cluster_{cluster}/cluster_{cluster}_chr{chrom}.msp.tsv"
    params:
        rfmix_out = rfmix_out_dir + "/cluster_{cluster}/cluster_{cluster}_chr{chrom}"
    resources:
        mem_mb=3000, time="00:20:00", cpus_per_task=2, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_envs/RFMix_env/bin:$PATH

        mkdir -p {rfmix_out_dir}/cluster_{wildcards.cluster}

        rfmix \
          -f {input.adm_vcf} \
          -r {input.ref_vcf} \
          -m {input.sample_map} \
          -g {recombmap_dir}/map_chr{wildcards.chrom} \
          -o {params.rfmix_out} \
          --chromosome={wildcards.chrom}
        """

# ================================================================
# CONSENSUS — SPECIFIC RULES
# ================================================================

rule generate_consensus:
    """Compare RFMix × GNOMix × AncestryHMM and produce genome-wide consensus."""
    input:
        done = expand(
            AncestryHMM_out_dir + "/cluster_{cluster}/chr{chrom}/ancestryHMM_chr{chrom}.done",
            chrom=CHROMS, allow_missing=True),
        msp_gnomix = expand(
            GNOMix_out_dir + "/cluster_{cluster}/cluster_{cluster}_chr{chrom}.msp",
            chrom=CHROMS, allow_missing=True),
        msp_rfmix = expand(
            rfmix_out_dir + "/cluster_{cluster}/cluster_{cluster}_chr{chrom}.msp.tsv",
            chrom=CHROMS, allow_missing=True)
    output:
        detailed_tsv = consensus_dir + "/cluster_{cluster}/cluster_{cluster}_genomewide_detailed_comparison.tsv",
        summary      = consensus_dir + "/cluster_{cluster}/cluster_{cluster}_concordance_summary.txt"
    params:
        script = script_consensus
    log:
        consensus_dir + "/cluster_{cluster}/logs/generate_consensus.log"
    resources:
        mem_mb=16000, time="00:60:00", cpus_per_task=1, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_envs/plot_env/bin:$PATH

        mkdir -p {consensus_dir}/cluster_{wildcards.cluster}/logs

        /shared/projects/admixture_yeast/LocAncInf/conda_envs/plot_env/bin/python {params.script} \
            --cluster cluster_{wildcards.cluster} \
            > {log} 2>&1
        """


rule consensus_phasing:
    """Phase consensus LAI regions using RFMix and GNOMix haplotype data."""
    input:
        detailed_tsv = consensus_dir + "/cluster_{cluster}/cluster_{cluster}_genomewide_detailed_comparison.tsv"
    output:
        phased_tsv = consensus_dir + "/cluster_{cluster}/cluster_{cluster}_genomewide_detailed_comparison_phased.tsv"
    params:
        script = script_phasing
    log:
        consensus_dir + "/cluster_{cluster}/logs/consensus_phasing.log"
    resources:
        mem_mb=20000, time="00:60:00", cpus_per_task=1, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_envs/plot_env/bin:$PATH

        mkdir -p {consensus_dir}/cluster_{wildcards.cluster}/logs

        /shared/projects/admixture_yeast/LocAncInf/conda_envs/plot_env/bin/python {params.script} \
            --cluster cluster_{wildcards.cluster} \
            > {log} 2>&1
        """

# ================================================================
# GLOBAL ANCESTRY — SPECIFIC RULES
# ================================================================

rule global_ancestry:
    """Compute per-sample global ancestry proportions from phased consensus TSV."""
    input:
        phased_tsv = consensus_dir + "/cluster_{cluster}/cluster_{cluster}_genomewide_detailed_comparison_phased.tsv"
    output:
        tsv = global_anc_dir + "/cluster_{cluster}/ancestry_proportions.tsv",
        png = global_anc_dir + "/cluster_{cluster}/ancestry_proportions_bar.svg"
    params:
        script = script_global_anc
    log:
        global_anc_dir + "/cluster_{cluster}/logs/global_ancestry.log"
    resources:
        mem_mb=2000, time="00:15:00", cpus_per_task=1, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_envs/plot_env/bin:$PATH

        mkdir -p {global_anc_dir}/cluster_{wildcards.cluster}/logs

        /shared/projects/admixture_yeast/LocAncInf/conda_envs/plot_env/bin/python {params.script} \
            --clusters {wildcards.cluster} \
            > {log} 2>&1
        """


# ================================================================
# TRACT LENGTH — SPECIFIC RULES
# ================================================================

rule tract_length_distribution:
    """Analyze phased ancestry tract lengths from consensus TSV."""
    input:
        phased_tsv = consensus_dir + "/cluster_{cluster}/cluster_{cluster}_genomewide_detailed_comparison_phased.tsv"
    output:
        hist    = tract_length_dir + "/cluster_{cluster}/tract_length_distribution.svg",
        barplot = tract_length_dir + "/cluster_{cluster}/per_sample_median_tractlength.svg"
    params:
        script = script_tract_length
    log:
        tract_length_dir + "/cluster_{cluster}/logs/tract_length_distribution.log"
    resources:
        mem_mb=2000, time="00:30:00", cpus_per_task=1, partition="fast"
    shell:
        """
        
        export PATH=/shared/projects/admixture_yeast/LocAncInf/conda_envs/plot_env/bin:$PATH

        mkdir -p {tract_length_dir}/cluster_{wildcards.cluster}/logs

        /shared/projects/admixture_yeast/LocAncInf/conda_envs/plot_env/bin/python {params.script} \
            --clusters {wildcards.cluster} \
            --nodata-gap 1000 \
            --unresolved-gap 2000 \
            > {log} 2>&1
        """
