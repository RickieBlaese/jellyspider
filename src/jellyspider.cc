#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <cstdint>

#include <dlfcn.h>
#include <gmp.h>

#include "mpreal/mpreal.h"


#ifndef WORK_SO_FILENAME
#define WORK_SO_FILENAME "libjellyspider-work.so"
#endif

#define STRINGIZE_NX(A) #A

#define STRINGIZE(A) STRINGIZE_NX(A)

#define VERSIONA 0
#define VERSIONB 1
#define VERSIONC 0
#define VERSION STRINGIZE(VERSIONA) "." STRINGIZE(VERSIONB) "." STRINGIZE(VERSIONC)


#ifdef DEBUG_BUILD

#define ERR_EXIT(A, ...) { \
    std::fprintf(stderr, "jspider: error: file " __FILE__ ":%i in %s(): ", __LINE__, __func__); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fputc('\n', stderr); \
    std::exit(static_cast<int>(A)); \
}

#define WARN(...) { \
    std::fprintf(stderr, "jspider: warning: file " __FILE__ ":%i in %s(): ", __LINE__, __func__); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fputc('\n', stderr); \
}

#else

#define ERR_EXIT(A, ...) { /* NOLINT */ \
    std::fputs("jspider: error: ", stderr); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fputc('\n', stderr); \
    std::exit(static_cast<int>(A)); \
}

#define WARN(...) { /* NOLINT */ \
    std::fputs("jspider: warning: ", stderr); \
    std::fprintf(stderr, __VA_ARGS__); \
    std::fputc('\n', stderr); \
}

#endif

template <typename T>
T get_random_int(const T &a, const T &b) {
    static std::random_device device{};
    static std::mt19937 engine(device());
    return std::uniform_int_distribution<>(a, b)(engine);
}

template <typename T>
T get_random_real(const T &a, const T &b) {
    static std::random_device device{};
    static std::mt19937 engine(device());
    return std::uniform_real_distribution<>(a, b)(engine);
}

gmp_randstate_t randstate; /* NOLINT */

/* NOLINTBEGIN */

/* even though we use a geometric distribution for generating mpzs, we need a reasonable upper limit for performance considerations (decide when to use) */
mp_bitcnt_t rand_geo_limit = 128;

/* NOLINTEND */

using dec_t = mpfr::mpreal;

static constexpr double geoadj = -0.0100503358535;

/* assumes out is initialized */
void mpz_geometric_dist(mpz_t &out) {
    mpz_t z;
    mpz_init(z);
    mpz_urandomb(z, randstate, rand_geo_limit);
    mpz_add_ui(z, z, 1); /* force > 0, also now can be 2^n so div by 2^n is perfect */
    dec_t a = z;
    mpfr_div_2ui(a.mpfr_ptr(), a.mpfr_srcptr(), rand_geo_limit, dec_t::get_default_rnd());
    a = mpfr::log(a);
    a /= geoadj;
    mpfr_get_z(out, a.mpfr_srcptr(), dec_t::get_default_rnd());
}

enum struct otype_t : std::uint16_t {
    none,

    /* literal */
    bool_v,
    /* literal */
    int_v,
    /* literal */
    dec_v,

    set, tuple,
    symbol,

    unresolvable
};

struct obj_t {
    std::optional<std::string> name;
    otype_t type = otype_t::none;
    std::variant<std::monostate, bool, mpz_t, dec_t> v; /* bool_v, int_v, and dec_v */

    bool operator==(const obj_t &other) const {
        bool res = type == other.type && name.has_value() == other.name.has_value() && v.index() == other.v.index();
        if (v.index() > 0 && v.index() == other.v.index()) {
            std::visit([&res, &other](const auto &val) {
                if constexpr (std::is_same_v<decltype(val), bool>) {
                    res = res && val == std::get<bool>(other.v);
                } else if constexpr (std::is_same_v<decltype(val), mpz_t>) {
                    res = res && !mpz_cmp(val, std::get<mpz_t>(other.v));
                } else if constexpr (std::is_same_v<decltype(val), dec_t>) {
                    res = res && val == std::get<dec_t>(other.v);
                }
            }, v);
        }
        if (name.has_value() && other.name.has_value()) {
            res = res && name.value() == other.name.value();
        }
        return res;
    }

    obj_t &operator=(const obj_t &other) { /* assumes v is std::monostate */
        name = other.name;
        type = other.type;
        std::visit([this](const auto &val) {
            if constexpr (std::is_same_v<decltype(val), mpz_t>) {
                v = mpz_t{};
                auto &tz = std::get<mpz_t>(v);
                mpz_init_set(tz, std::get<mpz_t>(val));
            } else {
                v = val;
            }
        }, other.v);
        return *this;
    }

    bool is_lit_obj() const {
        return (type == otype_t::bool_v || type == otype_t::int_v || type == otype_t::dec_v) && v.index() > 0;
    }
};

enum struct etype_t : std::uint16_t {
    none,

    objexpr_t,
    eqexpr_t,
    andexpr_t, orexpr_t, notexpr_t,
    unionexpr_t, intersectexpr_t, containedexpr_t, subsetexpr_t, subtractexpr_t,
    mapexpr_t
};

const std::unordered_map<etype_t, std::string> etype_to_str = { /* NOLINT */
    {etype_t::none, "none"},
    {etype_t::objexpr_t, "objexpr"},
    {etype_t::eqexpr_t, "eqexpr"},
    {etype_t::andexpr_t, "andexpr"},
    {etype_t::orexpr_t, "orexpr"},
    {etype_t::notexpr_t, "notexpr"},
    {etype_t::unionexpr_t, "unionexpr"},
    {etype_t::intersectexpr_t, "intersectexpr"},
    {etype_t::containedexpr_t, "containedexpr"},
    {etype_t::subsetexpr_t, "subsetexpr"},
    {etype_t::subtractexpr_t, "subtractexpr"},
    {etype_t::mapexpr_t, "mapexpr"}
};

struct expr_t;

using sptrexpr_t = std::shared_ptr<expr_t>;
using wptrexpr_t = std::weak_ptr<expr_t>;
using uptrexpr_t = std::unique_ptr<expr_t>;

struct thm_t;

using tid_t = std::uint64_t;

std::unordered_map<tid_t, thm_t> thms; /* NOLINT */

tid_t gen_thm_id() {
    static std::uint64_t counter = 0;
    return ++counter;
}

struct thm_t {
    tid_t id;
    std::string name;

    sptrexpr_t e; /* we assume e evaluates to a litobjexpr_t with obj holding true */

    explicit thm_t() : id(gen_thm_id()) {
        thms[id] = *this;
    }
};

struct pfstep_t {
    sptrexpr_t expr;

    tid_t id; /* id of the theorem used to get to expr */
};

/* std::vector<tid_t> find_thms(sr) {
} */

struct vec_t {
    std::vector<double> v;

    vec_t operator+(const vec_t &other) const {
        std::size_t sz = std::max(v.size(), other.v.size());
        std::size_t minsz = std::min(v.size(), other.v.size());
        vec_t r;
        r.v.resize(sz);
        for (std::size_t i = 0; i < minsz; i++) {
            r.v[i] = v[i] + other.v[i];
        }
        return r;
    }

    vec_t operator-(const vec_t &other) const {
        std::size_t sz = std::max(v.size(), other.v.size());
        std::size_t minsz = std::min(v.size(), other.v.size());
        vec_t r;
        r.v.resize(sz);
        for (std::size_t i = 0; i < minsz; i++) {
            r.v[i] = v[i] - other.v[i];
        }
        return r;
    }

    double dot(const vec_t &other) const {
        std::size_t minsz = std::min(v.size(), other.v.size());
        double s = 0;
        for (std::size_t i = 0; i < minsz; i++) {
            s += v[i] * other.v[i];
        }
        return s;
    }

    double euc_metric() const {
        double m = 0;
        for (const double &i : v) {
            m += i * i;
        }
        return std::sqrt(m);
    }

    double euc_distance(const vec_t &other) const {
        std::size_t minsz = std::min(v.size(), other.v.size());
        double m = 0;
        for (std::size_t i = 0; i < minsz; i++) {
            m += (v[i] - other.v[i]) * (v[i] - other.v[i]);
        }
        return std::sqrt(m);
    }
};

struct objexpr_t;


/* NOLINTBEGIN */
/* loaded with dlopen on startup */
uptrexpr_t (*expr_work)(const expr_t &e, const std::optional<sptrexpr_t> &target, const decltype(thms) &thms);
/* NOLINTEND */

struct expr_t { /* NOLINT */
    std::vector<sptrexpr_t> exprs;
    std::vector<wptrexpr_t> parents; /* destruct the locked sptrexpr_t from parents before destructing this expr */
    etype_t type = etype_t::none;
    uptrexpr_t work_cache, calc_cache; /* calc_cache doesn't hold a value when calculate failed */
    bool work_dirty = true, calc_dirty = true;

    explicit expr_t() = default;
    explicit expr_t(decltype(exprs) exprs, decltype(parents) parent) : exprs(std::move(exprs)), parents(std::move(parent)) {}
    virtual ~expr_t() = default;

    void clear() {
        for (sptrexpr_t &expr : exprs) {
            expr->clear();
        }
        exprs.clear();
    }


    /* walks */
    bool operator==(const expr_t &other) const;
    /* {
        bool res = type == other.type;
        res = res && obj.has_value() == other.obj.has_value();
        if (obj.has_value() && other.obj.has_value()) {
            res = res && obj.value() == other.obj.value();
        }
        for (std::uint32_t i = 0; i < exprs.size(); i++) {
            res = res && (*exprs[i] == *other.exprs[i]);
        }
        return res;
    } */

    /* walks (deep copy)
     * does not copy parent
     */
    virtual uptrexpr_t copy() const {
        uptrexpr_t e = std::make_unique<expr_t>();
        e->type = type;
        e->exprs.resize(exprs.size());
        for (std::size_t i = 0; i < exprs.size(); i++) {
            e->exprs[i] = exprs[i]->copy();
        }
        e->work_dirty = work_dirty;
        e->calc_dirty = calc_dirty;
        if (!work_dirty) {
            e->work_cache = work_cache->copy();
        }
        if (!calc_dirty) {
            e->calc_cache = calc_cache->copy();
        }
        return e;
    }

    void walk(const std::function<void (sptrexpr_t&)> &f) {
        for (sptrexpr_t &expr : exprs) {
            f(expr);
            expr->walk(f);
        }
    }

    void walk_const(const std::function<void (const sptrexpr_t&)> &f) const {
        for (const sptrexpr_t &expr : exprs) {
            f(expr);
            expr->walk_const(f);
        }
    }

    /* walks */
    std::uint32_t size() const {
        std::uint32_t sum = 1;
        walk_const([&sum](const sptrexpr_t &expr) { sum += expr->size() + 1; });
        return sum;
    }

    double dist(const expr_t &other, std::size_t sample_size) const;

    /* always assumes previous state was ok */
    void signal_dirty() {
        if (!work_dirty && !calc_dirty) {
            return;
        }
        work_dirty = true;
        calc_dirty = true;
        for (wptrexpr_t &parent : parents) {
            sptrexpr_t tmp = parent.lock();
            if (tmp) {
                tmp->signal_dirty();
            }
        }
    }

    /* if you want a redone work-ed expr that isn't from work_cache, just call expr_work */
    uptrexpr_t load(bool can_calculate) {
        if (can_calculate) {
            if (calc_dirty) {
                std::optional<uptrexpr_t> o = calculate();
                if (o.has_value()) {
                    calc_cache = std::move(o.value());
                }
                calc_dirty = false;
            }
            if (calc_cache) {
                return calc_cache->copy();
            }
        }
        if (work_dirty) {
            work_cache = expr_work(*this, {}, thms);
            work_dirty = false;
        }
        return work_cache->copy();
    }

    /* if optional has no value, it could not be calculated
     * otherwise should return an objexpr_t, with ALL (recursive) child exprs also objexpr_t - this is the only valid calcuated form (note we don't say "closed form" - it is more subjective and hazy, it depends on the context)
     * it is up to each derived expr_t class to determine precision requirements (DEAL WITH IT!!!)
     */
    virtual std::optional<uptrexpr_t> calculate() const;

    virtual uptrexpr_t generate() const;

    virtual std::string disp() const {
        return "_" + etype_to_str.at(type);
    }
};

/* copies */
template <typename T>
std::unique_ptr<T> from_expr_copy(const uptrexpr_t &e) {
    uptrexpr_t c = e->copy();
    return std::unique_ptr<T>(dynamic_cast<T*>(c.release()));
}

/* copies */
template <typename T>
std::shared_ptr<T> from_expr_copy(const sptrexpr_t &e) {
    uptrexpr_t c = e->copy();
    return std::unique_ptr<T>(dynamic_cast<T*>(c.release()));
}

/* moves (releases pointer) */
template <typename T>
uptrexpr_t to_expr(std::unique_ptr<T> &e) {
    return uptrexpr_t(dynamic_cast<expr_t*>(e.release()));
}

/* moves (releases pointer) */
template <typename T>
std::unique_ptr<T> from_expr(uptrexpr_t &e) {
    return std::unique_ptr<T>(dynamic_cast<T*>(e.release()));
}

struct objexpr_t : virtual expr_t {
    obj_t obj;

    uptrexpr_t generate() const override;

    uptrexpr_t copy() const override;

    ~objexpr_t() override {
        if (obj.type == otype_t::int_v) {
            mpz_clear(std::get<mpz_t>(obj.v));
        }
    }

    /* assumes obj types match, and is_lit_obj */
    double obj_dist(const objexpr_t &other) {
        switch (obj.type) {
            case otype_t::bool_v:
                return static_cast<double>(std::get<bool>(obj.v) == std::get<bool>(other.obj.v));
            case otype_t::int_v:
                return 1;
                AAAAAAA;
        }
    }
};

using sptrobjexpr_t = std::shared_ptr<objexpr_t>;
using uptrobjexpr_t = std::unique_ptr<objexpr_t>;

uptrexpr_t objexpr_t::copy() const {
    uptrexpr_t e = expr_t::copy();
    uptrobjexpr_t oe = from_expr<objexpr_t>(e);
    oe->obj = obj;
    return to_expr(oe);
}

uptrobjexpr_t make_bool(bool v) {
    uptrobjexpr_t oe = std::make_unique<objexpr_t>();
    oe->obj = obj_t{.type = otype_t::bool_v, .v = v};
    return oe;
}

uptrobjexpr_t make_mpz(mpz_t i) {
    uptrobjexpr_t oe = std::make_unique<objexpr_t>();
    oe->obj = obj_t{.type = otype_t::int_v, .v = mpz_t{}};
    mpz_t &tz = std::get<mpz_t>(oe->obj.v); /* NOLINT */
    mpz_init_set(tz, i);
    return oe;
}

uptrobjexpr_t make_dec(const dec_t &a) {
    uptrobjexpr_t oe = std::make_unique<objexpr_t>();
    oe->obj = obj_t{.type = otype_t::dec_v, .v = a};
    return oe;
}

/* this is hard... for non lit objs */
uptrexpr_t objexpr_t::generate() const {
    switch (obj.type) {
        case otype_t::bool_v:
            return make_bool(get_random_int<bool>(false, true));
        case otype_t::int_v:
            {
                mpz_t a;
                mpz_init(a);
                mpz_geometric_dist(a);
                return make_mpz(a);
            }
        case otype_t::dec_v:
            {
                dec_t a;
                mpfr_urandom(a.mpfr_ptr(), randstate, dec_t::get_default_rnd());
                return make_dec(a);
            }
        case otype_t::unresolvable:
        case otype_t::none:

        /* TODO(stole): work on these !! hard :( */
        case otype_t::symbol:
        case otype_t::set:
        case otype_t::tuple:
            return copy();
    }
}

double expr_t::dist(const expr_t &other, std::size_t sample_size = 20) const {
    sptrexpr_t e = copy(), o = other.copy();
    /* first we collect all objs */
    std::vector<sptrobjexpr_t> all_objs;
    e->walk_const([&all_objs](const sptrexpr_t &texpr) {
        if (texpr->type == etype_t::objexpr_t && texpr->exprs.empty()) {
            sptrobjexpr_t tobjexpr = std::static_pointer_cast<objexpr_t>(texpr);
            if (tobjexpr->obj.name.has_value()) {
                all_objs.push_back(tobjexpr);
            }
        }
    });
    std::vector<sptrobjexpr_t> all_objs_o;
    o->walk_const([&all_objs_o](const sptrexpr_t &texpr) {
        if (texpr->type == etype_t::objexpr_t && texpr->exprs.empty()) {
            sptrobjexpr_t tobjexpr = std::static_pointer_cast<objexpr_t>(texpr);
            if (tobjexpr->obj.name.has_value()) {
                all_objs_o.push_back(tobjexpr);
            }
        }
    });
    std::sort(all_objs.begin(), all_objs.end(), [](const sptrobjexpr_t &a, const sptrobjexpr_t &b) {
        return a->obj.name.value() < b->obj.name.value();
    });
    std::sort(all_objs_o.begin(), all_objs_o.end(), [](const sptrobjexpr_t &a, const sptrobjexpr_t &b) {
        return a->obj.name.value() < b->obj.name.value();
    });
    if (all_objs != all_objs_o) {

    }
    for (std::size_t j = 0; j < sample_size; j++) {
        vec_t ev, ov;
        ev.v.resize(all_objs.size());
        for (sptrobjexpr_t &so : all_objs) {
            uptrexpr_t gen = so->generate();
            std::unique_ptr<objexpr_t> uo = from_expr<objexpr_t>(gen);
            so->obj = uo->obj;
            so->signal_dirty();
        }

    }
}

int main() {
    gmp_randinit_default(randstate);

    void *dlhandle = dlopen(WORK_SO_FILENAME, RTLD_LAZY);
    expr_work = reinterpret_cast<decltype(expr_work)>(dlsym(dlhandle, "expr_work")); /* NOLINT */
    if (expr_work == nullptr) {
        ERR_EXIT(1, "could not resolve expr_work from shared object " WORK_SO_FILENAME);
    }
    dlclose(dlhandle);
}
