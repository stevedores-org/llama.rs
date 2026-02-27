use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AgentRole {
    Planner,
    Coder,
    Reviewer,
    Tester,
    Fixer,
}

impl AgentRole {
    fn precedence(self) -> usize {
        match self {
            AgentRole::Planner => 0,
            AgentRole::Coder => 1,
            AgentRole::Reviewer => 2,
            AgentRole::Tester => 3,
            AgentRole::Fixer => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArtifactKind {
    Requirement,
    Plan,
    Patch,
    TestReport,
    Review,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    pub kind: ArtifactKind,
    pub title: String,
    pub body: String,
    pub fields: BTreeMap<String, String>,
}

impl Artifact {
    pub fn new(kind: ArtifactKind, title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            kind,
            title: title.into(),
            body: body.into(),
            fields: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandoffContract {
    pub from: AgentRole,
    pub to: AgentRole,
    pub input_kind: ArtifactKind,
    pub output_kind: ArtifactKind,
    pub required_fields: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractValidationError {
    WrongInputKind {
        expected: ArtifactKind,
        found: ArtifactKind,
    },
    WrongOutputKind {
        expected: ArtifactKind,
        found: ArtifactKind,
    },
    MissingField {
        field: String,
    },
}

pub fn validate_handoff(
    contract: &HandoffContract,
    input: &Artifact,
    output: &Artifact,
) -> Result<(), ContractValidationError> {
    if input.kind != contract.input_kind {
        return Err(ContractValidationError::WrongInputKind {
            expected: contract.input_kind,
            found: input.kind,
        });
    }
    if output.kind != contract.output_kind {
        return Err(ContractValidationError::WrongOutputKind {
            expected: contract.output_kind,
            found: output.kind,
        });
    }
    for required in &contract.required_fields {
        if !output.fields.contains_key(required) {
            return Err(ContractValidationError::MissingField {
                field: required.clone(),
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrchestrationTask {
    pub id: String,
    pub role: AgentRole,
    pub depends_on: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingError {
    UnknownDependency { task_id: String, dependency: String },
    CycleDetected,
}

pub fn dependency_aware_route(tasks: &[OrchestrationTask]) -> Result<Vec<String>, RoutingError> {
    let mut task_ids = BTreeSet::new();
    for task in tasks {
        task_ids.insert(task.id.clone());
    }
    for task in tasks {
        for dep in &task.depends_on {
            if !task_ids.contains(dep) {
                return Err(RoutingError::UnknownDependency {
                    task_id: task.id.clone(),
                    dependency: dep.clone(),
                });
            }
        }
    }

    let mut indegree: BTreeMap<String, usize> = BTreeMap::new();
    let mut edges: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for task in tasks {
        indegree.entry(task.id.clone()).or_insert(0);
        edges.entry(task.id.clone()).or_default();
    }

    for task in tasks {
        for dep in &task.depends_on {
            *indegree
                .get_mut(&task.id)
                .expect("task id inserted into indegree") += 1;
            edges.entry(dep.clone()).or_default().push(task.id.clone());
        }
    }

    for next in edges.values_mut() {
        next.sort();
    }

    let mut ready: BTreeSet<String> = indegree
        .iter()
        .filter_map(|(id, in_deg)| if *in_deg == 0 { Some(id.clone()) } else { None })
        .collect();

    let mut ordered = Vec::with_capacity(tasks.len());
    while let Some(id) = ready.first().cloned() {
        ready.remove(&id);
        ordered.push(id.clone());
        if let Some(nexts) = edges.get(&id) {
            for next in nexts {
                let entry = indegree
                    .get_mut(next)
                    .expect("edge points to known task id");
                *entry -= 1;
                if *entry == 0 {
                    ready.insert(next.clone());
                }
            }
        }
    }

    if ordered.len() != tasks.len() {
        return Err(RoutingError::CycleDetected);
    }
    Ok(ordered)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PatchOutput {
    pub role: AgentRole,
    pub task_id: String,
    pub target: String,
    pub patch: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeConflict {
    pub target: String,
    pub chosen_task_id: String,
    pub competing_task_ids: Vec<String>,
    pub explanation: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeResult {
    pub merged: BTreeMap<String, String>,
    pub conflicts: Vec<MergeConflict>,
}

pub fn merge_role_outputs(outputs: &[PatchOutput]) -> MergeResult {
    let mut by_target: BTreeMap<String, Vec<&PatchOutput>> = BTreeMap::new();
    for output in outputs {
        by_target
            .entry(output.target.clone())
            .or_default()
            .push(output);
    }

    let mut merged = BTreeMap::new();
    let mut conflicts = Vec::new();

    for (target, mut candidates) in by_target {
        candidates.sort_by(|left, right| {
            right
                .role
                .precedence()
                .cmp(&left.role.precedence())
                .then_with(|| left.task_id.cmp(&right.task_id))
        });

        let winner = candidates[0];
        merged.insert(target.clone(), winner.patch.clone());

        let mut competing = Vec::new();
        for other in candidates.iter().skip(1) {
            if other.patch != winner.patch {
                competing.push(other.task_id.clone());
            }
        }

        if !competing.is_empty() {
            conflicts.push(MergeConflict {
                target,
                chosen_task_id: winner.task_id.clone(),
                competing_task_ids: competing,
                explanation: "deterministic precedence by role (Fixer > Tester > Reviewer > Coder > Planner), then task id".to_string(),
            });
        }
    }

    MergeResult { merged, conflicts }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkflowNode {
    pub task: OrchestrationTask,
    pub contract: Option<HandoffContract>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowError {
    Routing(RoutingError),
    MissingInput { task_id: String },
    MissingExecutor { role: AgentRole },
    Validation(ContractValidationError),
}

pub type RoleExecutor = fn(&Artifact) -> Artifact;

pub fn run_multi_role_workflow(
    nodes: &[WorkflowNode],
    initial: Artifact,
    executors: &HashMap<AgentRole, RoleExecutor>,
) -> Result<BTreeMap<String, Artifact>, WorkflowError> {
    let route = dependency_aware_route(
        &nodes
            .iter()
            .map(|node| node.task.clone())
            .collect::<Vec<_>>(),
    )
    .map_err(WorkflowError::Routing)?;

    let mut by_id: HashMap<&str, &WorkflowNode> = HashMap::new();
    for node in nodes {
        by_id.insert(&node.task.id, node);
    }

    let mut produced: BTreeMap<String, Artifact> = BTreeMap::new();

    for task_id in route {
        let node = by_id
            .get(task_id.as_str())
            .expect("route only contains declared tasks");

        let input = if node.task.depends_on.is_empty() {
            initial.clone()
        } else {
            let first_dep = node.task.depends_on[0].as_str();
            produced
                .get(first_dep)
                .cloned()
                .ok_or_else(|| WorkflowError::MissingInput {
                    task_id: node.task.id.clone(),
                })?
        };

        let executor = executors
            .get(&node.task.role)
            .ok_or(WorkflowError::MissingExecutor {
                role: node.task.role,
            })?;
        let output = executor(&input);

        if let Some(contract) = &node.contract {
            validate_handoff(contract, &input, &output).map_err(WorkflowError::Validation)?;
        }

        produced.insert(node.task.id.clone(), output);
    }

    Ok(produced)
}

pub fn run_requirement_to_patch_example(requirement: &str) -> Result<MergeResult, WorkflowError> {
    let nodes = vec![
        WorkflowNode {
            task: OrchestrationTask {
                id: "plan".to_string(),
                role: AgentRole::Planner,
                depends_on: Vec::new(),
            },
            contract: Some(HandoffContract {
                from: AgentRole::Planner,
                to: AgentRole::Coder,
                input_kind: ArtifactKind::Requirement,
                output_kind: ArtifactKind::Plan,
                required_fields: vec!["tasks".to_string()],
            }),
        },
        WorkflowNode {
            task: OrchestrationTask {
                id: "code".to_string(),
                role: AgentRole::Coder,
                depends_on: vec!["plan".to_string()],
            },
            contract: Some(HandoffContract {
                from: AgentRole::Coder,
                to: AgentRole::Reviewer,
                input_kind: ArtifactKind::Plan,
                output_kind: ArtifactKind::Patch,
                required_fields: vec!["target".to_string()],
            }),
        },
        WorkflowNode {
            task: OrchestrationTask {
                id: "review".to_string(),
                role: AgentRole::Reviewer,
                depends_on: vec!["code".to_string()],
            },
            contract: Some(HandoffContract {
                from: AgentRole::Reviewer,
                to: AgentRole::Fixer,
                input_kind: ArtifactKind::Patch,
                output_kind: ArtifactKind::Review,
                required_fields: vec!["target".to_string()],
            }),
        },
        WorkflowNode {
            task: OrchestrationTask {
                id: "test".to_string(),
                role: AgentRole::Tester,
                depends_on: vec!["code".to_string()],
            },
            contract: Some(HandoffContract {
                from: AgentRole::Tester,
                to: AgentRole::Fixer,
                input_kind: ArtifactKind::Patch,
                output_kind: ArtifactKind::TestReport,
                required_fields: vec!["status".to_string()],
            }),
        },
        WorkflowNode {
            task: OrchestrationTask {
                id: "fix".to_string(),
                role: AgentRole::Fixer,
                depends_on: vec!["review".to_string(), "test".to_string()],
            },
            contract: Some(HandoffContract {
                from: AgentRole::Fixer,
                to: AgentRole::Fixer,
                input_kind: ArtifactKind::Review,
                output_kind: ArtifactKind::Patch,
                required_fields: vec!["target".to_string()],
            }),
        },
    ];

    fn planner(input: &Artifact) -> Artifact {
        let mut artifact = Artifact::new(
            ArtifactKind::Plan,
            "plan",
            format!(
                "Break requirement into implementation and test tasks: {}",
                input.body
            ),
        );
        artifact
            .fields
            .insert("tasks".to_string(), "implement,verify".to_string());
        artifact
    }

    fn coder(_input: &Artifact) -> Artifact {
        let mut artifact = Artifact::new(
            ArtifactKind::Patch,
            "initial patch",
            "diff --git a/src/lib.rs b/src/lib.rs\n+pub fn feature() -> bool { false }",
        );
        artifact
            .fields
            .insert("target".to_string(), "src/lib.rs".to_string());
        artifact
    }

    fn reviewer(input: &Artifact) -> Artifact {
        let mut artifact = Artifact::new(
            ArtifactKind::Review,
            "review notes",
            format!("Need deterministic behavior for patch: {}", input.title),
        );
        let target = input
            .fields
            .get("target")
            .cloned()
            .unwrap_or_else(|| "src/lib.rs".to_string());
        artifact.fields.insert("target".to_string(), target);
        artifact
    }

    fn tester(_input: &Artifact) -> Artifact {
        let mut artifact = Artifact::new(
            ArtifactKind::TestReport,
            "test report",
            "all targeted tests pass",
        );
        artifact
            .fields
            .insert("status".to_string(), "pass".to_string());
        artifact
    }

    fn fixer(input: &Artifact) -> Artifact {
        let mut artifact = Artifact::new(
            ArtifactKind::Patch,
            "final patch",
            format!(
                "diff --git a/src/lib.rs b/src/lib.rs\n+pub fn feature() -> bool {{ true }}\n# reason: {}",
                input.body
            ),
        );
        let target = input
            .fields
            .get("target")
            .cloned()
            .unwrap_or_else(|| "src/lib.rs".to_string());
        artifact.fields.insert("target".to_string(), target);
        artifact
    }

    let mut executors: HashMap<AgentRole, RoleExecutor> = HashMap::new();
    executors.insert(AgentRole::Planner, planner);
    executors.insert(AgentRole::Coder, coder);
    executors.insert(AgentRole::Reviewer, reviewer);
    executors.insert(AgentRole::Tester, tester);
    executors.insert(AgentRole::Fixer, fixer);

    let initial = Artifact::new(ArtifactKind::Requirement, "requirement", requirement);
    let produced = run_multi_role_workflow(&nodes, initial, &executors)?;

    let mut outputs = Vec::new();
    if let Some(code) = produced.get("code") {
        outputs.push(PatchOutput {
            role: AgentRole::Coder,
            task_id: "code".to_string(),
            target: code
                .fields
                .get("target")
                .cloned()
                .unwrap_or_else(|| "src/lib.rs".to_string()),
            patch: code.body.clone(),
        });
    }
    if let Some(fix) = produced.get("fix") {
        outputs.push(PatchOutput {
            role: AgentRole::Fixer,
            task_id: "fix".to_string(),
            target: fix
                .fields
                .get("target")
                .cloned()
                .unwrap_or_else(|| "src/lib.rs".to_string()),
            patch: fix.body.clone(),
        });
    }

    Ok(merge_role_outputs(&outputs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handoff_validator_rejects_wrong_kind_and_missing_fields() {
        let contract = HandoffContract {
            from: AgentRole::Planner,
            to: AgentRole::Coder,
            input_kind: ArtifactKind::Requirement,
            output_kind: ArtifactKind::Plan,
            required_fields: vec!["tasks".to_string()],
        };

        let input = Artifact::new(ArtifactKind::Requirement, "req", "need feature");
        let wrong_output = Artifact::new(ArtifactKind::Patch, "bad", "oops");
        let wrong = validate_handoff(&contract, &input, &wrong_output).unwrap_err();
        assert!(matches!(
            wrong,
            ContractValidationError::WrongOutputKind { .. }
        ));

        let output_without_field = Artifact::new(ArtifactKind::Plan, "plan", "items");
        let missing = validate_handoff(&contract, &input, &output_without_field).unwrap_err();
        assert!(matches!(
            missing,
            ContractValidationError::MissingField { .. }
        ));
    }

    #[test]
    fn dependency_routing_is_deterministic_and_cycle_safe() {
        let tasks = vec![
            OrchestrationTask {
                id: "fix".to_string(),
                role: AgentRole::Fixer,
                depends_on: vec!["review".to_string(), "test".to_string()],
            },
            OrchestrationTask {
                id: "review".to_string(),
                role: AgentRole::Reviewer,
                depends_on: vec!["code".to_string()],
            },
            OrchestrationTask {
                id: "test".to_string(),
                role: AgentRole::Tester,
                depends_on: vec!["code".to_string()],
            },
            OrchestrationTask {
                id: "code".to_string(),
                role: AgentRole::Coder,
                depends_on: vec!["plan".to_string()],
            },
            OrchestrationTask {
                id: "plan".to_string(),
                role: AgentRole::Planner,
                depends_on: Vec::new(),
            },
        ];

        let route = dependency_aware_route(&tasks).unwrap();
        assert_eq!(route, vec!["plan", "code", "review", "test", "fix"]);

        let cyc = vec![
            OrchestrationTask {
                id: "a".to_string(),
                role: AgentRole::Planner,
                depends_on: vec!["b".to_string()],
            },
            OrchestrationTask {
                id: "b".to_string(),
                role: AgentRole::Coder,
                depends_on: vec!["a".to_string()],
            },
        ];
        assert!(matches!(
            dependency_aware_route(&cyc),
            Err(RoutingError::CycleDetected)
        ));
    }

    #[test]
    fn deterministic_merge_reports_conflicts() {
        let outputs = vec![
            PatchOutput {
                role: AgentRole::Coder,
                task_id: "code".to_string(),
                target: "src/lib.rs".to_string(),
                patch: "coder patch".to_string(),
            },
            PatchOutput {
                role: AgentRole::Fixer,
                task_id: "fix".to_string(),
                target: "src/lib.rs".to_string(),
                patch: "fixer patch".to_string(),
            },
            PatchOutput {
                role: AgentRole::Tester,
                task_id: "test".to_string(),
                target: "tests/e2e.rs".to_string(),
                patch: "test patch".to_string(),
            },
        ];

        let merged = merge_role_outputs(&outputs);
        assert_eq!(
            merged.merged.get("src/lib.rs").map(String::as_str),
            Some("fixer patch")
        );
        assert_eq!(
            merged.merged.get("tests/e2e.rs").map(String::as_str),
            Some("test patch")
        );
        assert_eq!(merged.conflicts.len(), 1);
        assert_eq!(merged.conflicts[0].target, "src/lib.rs");
        assert_eq!(merged.conflicts[0].chosen_task_id, "fix");
    }

    #[test]
    fn multi_role_workflow_is_reproducible_in_ci() {
        let run1 = run_requirement_to_patch_example("add deterministic role orchestration")
            .expect("run 1 should succeed");
        let run2 = run_requirement_to_patch_example("add deterministic role orchestration")
            .expect("run 2 should succeed");

        assert_eq!(run1, run2);
        assert_eq!(
            run1.merged.get("src/lib.rs").map(String::as_str),
            Some(
                "diff --git a/src/lib.rs b/src/lib.rs\n+pub fn feature() -> bool { true }\n# reason: Need deterministic behavior for patch: initial patch"
            )
        );
        assert_eq!(run1.conflicts.len(), 1);
    }
}
