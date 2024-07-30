struct EquationData <: AbstractVarEqMapping
    depvarmap::Any
    indvarmap::Any
    args::Any
    ivargs::Any
    argmap::Any
end

function EquationData(pdesys, varmap, strategy)
    eqs = map(eq -> eq.lhs, pdesys.eqs)
    bcs = map(eq -> eq.lhs, pdesys.bcs)
    alleqs = vcat(eqs, bcs)

    argmap = map(alleqs) do eq
        eq => get_argument([eq], varmap)[1]
    end |> Dict

    depvarmap = map(alleqs) do eq
        eq => get_depvars(eq, varmap.depvar_ops)
    end |> Dict

    indvarmap = map(alleqs) do eq
        eq => get_indvars(eq, varmap)
    end |> Dict

    # Why?
    if strategy isa QuadratureTraining
        _args = get_argument(alleqs, varmap)
    else
        _args = get_variables(alleqs, varmap)
    end

    args = map(zip(alleqs, _args)) do (eq, args)
        eq => args
    end |> Dict

    ivargs = get_iv_argument(alleqs, varmap)

    ivargs = map(zip(alleqs, ivargs)) do (eq, args)
        eq => args
    end |> Dict

    EquationData(depvarmap, indvarmap, args, ivargs, argmap)
end

function depvars(eq, eqdata::EquationData)
    eqdata.depvarmap[eq]
end

function indvars(eq, eqdata::EquationData)
    eqdata.indvarmap[eq]
end

function eq_args(eq, eqdata::EquationData)
    eqdata.args[eq]
end

function eq_iv_args(eq, eqdata::EquationData)
    eqdata.ivargs[eq]
end

argument(eq, eqdata) = eqdata.argmap[eq]

function get_iv_argument(eqs, v::VariableMap)
    vars = map(eqs) do eq
        _vars = map(depvar -> get_depvars(eq, [depvar]), v.depvar_ops)
        f_vars = filter(x -> !isempty(x), _vars)
        mapreduce(vars -> mapreduce(op -> v.args[op], vcat, operation.(vars), init = []),
            vcat, f_vars, init = [])
    end
    args_ = map(vars) do _vars
        seen = []
        filter(_vars) do x
            if x isa Number
                error("Unreachable")
            else
                if any(isequal(x), seen)
                    false
                else
                    push!(seen, x)
                    true
                end
            end
        end
    end
    return args_
end

"""
    get_iv_variables(eqs, v::VariableMap)

Returns all variables that are used in each equations or boundary condition.
"""
function get_iv_variables(eqs, v::VariableMap)
    args = get_iv_argument(eqs, v)
    return map(arg -> filter(x -> !(x isa Number), arg), args)
end