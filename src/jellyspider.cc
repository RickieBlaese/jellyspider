#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <type_traits>
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

template <typename T>
using Inout = std::type_identity_t<T>;

template <typename T>
using Out = std::type_identity_t<T>;

#define Inout(T) Inout<T>
#define Out(T) Out<T>

#define DEF_ALL_PTRS(X) using sptr ## X = std::shared_ptr<X>; \
using wptr ## X = std::weak_ptr<X>; \
using uptr ## X = std::unique_ptr<X>;


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

using dec_t = mpfr::mpreal;

struct mpzw_t {
    mpz_t z;
};

/* NOLINTBEGIN */

gmp_randstate_t randstate;

/* even though we use a geometric distribution for generating mpzs, we need a reasonable upper limit for performance considerations (decide when to use) */
mp_bitcnt_t rand_geo_limit = 128;

#define GEN_DEFAULT_BASE 0.9
#define DIST_CHCK_SUBSET_MAX_SIZE 100

/* NOLINTEND */

/* assumes out is initialized */
void mpzs_geometric_dist(mpzw_t &out, const dec_t &base = GEN_DEFAULT_BASE) {
    dec_t a(0, static_cast<mpfr_prec_t>(rand_geo_limit));
    mpfr_urandomb(a.mpfr_ptr(), randstate); /* now a is in [0, 1) */
    a = mpfr::log(1.0 - a);
    a /= mpfr::log(base);
    mpfr_sqr(a.mpfr_ptr(), a.mpfr_srcptr(), dec_t::get_default_rnd());
    mpfr_get_z(out.z, a.mpfr_srcptr(), dec_t::get_default_rnd());
}

void dec_compress(dec_t &a, const dec_t &base = GEN_DEFAULT_BASE) {
    mpfr_sqrt(a.mpfr_ptr(), a.mpfr_srcptr(), dec_t::get_default_rnd());
    mpfr_pow(a.mpfr_ptr(), base.mpfr_srcptr(), a.mpfr_srcptr(), dec_t::get_default_rnd());
    mpfr_ui_sub(a.mpfr_ptr(), 1, a.mpfr_srcptr(), dec_t::get_default_rnd());
}


double double_compress(double a, const double &base = GEN_DEFAULT_BASE) {
    a = std::sqrt(a);
    a = std::pow(base, a);
    return 1 - a;
}

/* assumes a > 0 */
double mpzs_compress_double(const mpzw_t &a, const dec_t &base = GEN_DEFAULT_BASE) {
    dec_t r = a.z;
    dec_compress(r, base);
    return r.toDouble();
}

struct range_t {
    std::size_t a = 0, b = 0;
};

struct matched_pair_t {
    range_t x, y;
};

template <typename T>
double vec_similarity(const std::vector<T> &x, const std::vector<T> &y) {
    std::vector<matched_pair_t> s;
    for (std::size_t ai = 0; ai < x.size(); ai++) {
        for (std::size_t bi = 0; bi < y.size(); bi++) {
            if (x[ai] == y[bi]) {

                /* check for overlap */
                bool skip = false;
                for (const matched_pair_t &mp : s) {
                     if ((mp.x.a <= ai && ai < mp.x.b) || (mp.y.a <= bi && bi < mp.y.b)) {
                        skip = true;
                        break;
                    }
                }
                if (skip) {
                    continue;
                }

                s.push_back(matched_pair_t{range_t{ai, ai+1}, range_t{bi, bi+1}});
                for (ai++, bi++; ai < x.size() && bi < y.size(); ai++, bi++) {
                    if (x[ai] != y[bi]) {
                        break;
                    }
                }
                s[s.size() - 1].x.b = ai;
                s[s.size() - 1].y.b = bi;
            }
        }
    }

    std::size_t total_len = 0;
    for (const matched_pair_t &mp : s) {
        total_len += mp.x.b - mp.x.a;
    }

    return static_cast<double>(total_len)/static_cast<double>(std::min(x.size(), y.size()));
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
    std::optional<std::string> name; /* has value when it's an unspecified object - i.e. with .v as monostate */
    otype_t type = otype_t::none;
    std::variant<std::monostate, bool, mpzw_t, dec_t> v; /* bool_v, int_v, and dec_v */

    bool operator==(const obj_t &other) const {
        bool res = type == other.type && name.has_value() == other.name.has_value() && v.index() == other.v.index();
        if (v.index() > 0 && v.index() == other.v.index()) {
            std::visit([&res, &other](const auto &val) {
                if constexpr (std::is_same_v<decltype(val), bool>) {
                    res = res && val == std::get<bool>(other.v);
                } else if constexpr (std::is_same_v<decltype(val), mpzw_t>) {
                    res = res && !mpz_cmp(val, std::get<mpzw_t>(other.v));
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
            if constexpr (std::is_same_v<decltype(val), mpzw_t>) {
                v = mpzw_t{};
                auto &tz = std::get<mpzw_t>(v);
                mpz_init_set(tz, std::get<mpzw_t>(val));
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
    addexpr_t,
    eqexpr_t,
    andexpr_t, orexpr_t, notexpr_t,
    unionexpr_t, intersectexpr_t, containedexpr_t, subsetexpr_t, subtractexpr_t,
    stexpr_t,
    setgenexpr_t,
    mapexpr_t
};

const std::unordered_map<etype_t, std::string> etype_to_str = { /* NOLINT */
    {etype_t::none, "none"},
    {etype_t::objexpr_t, "objexpr"},
    {etype_t::addexpr_t, "addexpr"},
    {etype_t::eqexpr_t, "eqexpr"},
    {etype_t::andexpr_t, "andexpr"},
    {etype_t::orexpr_t, "orexpr"},
    {etype_t::notexpr_t, "notexpr"},
    {etype_t::unionexpr_t, "unionexpr"},
    {etype_t::intersectexpr_t, "intersectexpr"},
    {etype_t::containedexpr_t, "containedexpr"},
    {etype_t::subsetexpr_t, "subsetexpr"},
    {etype_t::subtractexpr_t, "subtractexpr"},
    {etype_t::stexpr_t, "stexpr"},
    {etype_t::setgenexpr_t, "setgenexpr"},
    {etype_t::mapexpr_t, "mapexpr"}
};

struct expr_t;

DEF_ALL_PTRS(expr_t);

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

DEF_ALL_PTRS(objexpr_t)

/* NOLINTBEGIN */

/* loaded with dlopen on startup
 * can return an empty uptrexpr_t if it wasn't going to change it */
uptrexpr_t (*expr_work)(const expr_t &e, const std::optional<sptrexpr_t> &target, const decltype(thms) &thms);

/* NOLINTEND */

struct expr_t { /* NOLINT */
    std::vector<sptrexpr_t> exprs;
    std::vector<wptrexpr_t> parents; /* destruct the locked sptrexpr_t from parents before destructing this expr */
    etype_t type = etype_t::none;
    uptrexpr_t work_cache, calc_cache; /* calc_cache doesn't hold a value when calculate failed */
    bool work_dirty = true, calc_dirty = true;

    explicit expr_t() = default;
    explicit expr_t(decltype(exprs) exprs, decltype(parents) parents) : exprs(std::move(exprs)), parents(std::move(parents)) {}
    virtual ~expr_t() = default;

    void clear() {
        for (sptrexpr_t &expr : exprs) {
            expr->clear();
        }
        exprs.clear();
    }

    bool is_calc_form() const {
        if (type != etype_t::objexpr_t) { return false; }
        bool calc_form = true;
        for (const sptrexpr_t &expr : exprs) {
            calc_form = calc_form && expr->is_calc_form();
        }
        return calc_form;
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

    template <typename T>
    uptrexpr_t base_copy_shallow() const {
        uptrexpr_t e(dynamic_cast<expr_t*>(new T));
        e->type = type;
        e->exprs = exprs;
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

    virtual uptrexpr_t copy_shallow() const {
        return base_copy_shallow<expr_t>();
    }

    /* walks (deep copy)
     * does not copy parent
     */
    template <typename T>
    uptrexpr_t base_copy() const {
        uptrexpr_t e(dynamic_cast<expr_t*>(new T));
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

    virtual uptrexpr_t copy() const {
        return base_copy<expr_t>();
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

    void get_deepest_type_stacks(std::vector<std::vector<etype_t>> &out, std::vector<etype_t> &cur) const {
        cur.push_back(type);
        if (exprs.empty()) {
            out.push_back(cur);
            return;
        }
        for (const sptrexpr_t &e : exprs) {
            e->get_deepest_type_stacks(out, cur);
        }
        cur.erase(cur.end() - 1);
    }

    /* walks */
    std::uint32_t size() const {
        std::uint32_t sum = 1;
        walk_const([&sum](const sptrexpr_t &expr) { sum += expr->size() + 1; });
        return sum;
    }

    double dist_randgen(const expr_t &other, std::size_t sample_size) const;

    double dist_match(const expr_t &other) const;

    /* dist are between 0 and 1 */
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
    virtual std::optional<uptrexpr_t> calculate() const {
        return {};
    }

    virtual uptrexpr_t generate() const {
        return copy();
    }

    virtual std::string disp() const {
        return "_" + etype_to_str.at(type);
    }
};

void expr_add_expr(const sptrexpr_t &parent, const sptrexpr_t &expr) {
    expr->parents.push_back(parent);
    parent->exprs.push_back(expr);
}


/* copies */
template <typename T>
std::unique_ptr<T> from_expr_copy(const uptrexpr_t &e) {
    uptrexpr_t c = e->copy();
    return std::unique_ptr<T>(static_cast<T*>(c.release()));
}

/* copies */
template <typename T>
std::shared_ptr<T> from_expr_copy(const sptrexpr_t &e) {
    uptrexpr_t c = e->copy();
    return std::unique_ptr<T>(static_cast<T*>(c.release()));
}

/* moves (releases pointer) */
template <typename T>
uptrexpr_t to_expr(std::unique_ptr<T> &e) {
    return uptrexpr_t(dynamic_cast<expr_t*>(e.release()));
}

/* moves (releases pointer) */
template <typename T>
std::unique_ptr<T> from_expr(uptrexpr_t &e) {
    auto t1 = e.release();
    auto t2 = dynamic_cast<T*>(t1);
    return std::unique_ptr<T>(t2);
}


struct objexpr_t : virtual expr_t {
    obj_t obj;

    uptrexpr_t generate() const override;

    uptrexpr_t copy_shallow() const override;
    uptrexpr_t copy() const override;

    objexpr_t() {
        type = etype_t::objexpr_t;
    }

    ~objexpr_t() override {
        if (obj.type == otype_t::int_v) {
            mpz_clear(std::get<mpzw_t>(obj.v).z);
        }
    }

    std::optional<uptrexpr_t> calculate() const override {
        if (obj.name.has_value() && obj.v.index() == 0) {
            return {};
        }
        return copy();
    }

    /* dist are between 0 and 1 */
    /* assumes obj types match, and this expr is_calc_form */
    double obj_dist(const objexpr_t &other) {
        switch (obj.type) {
            case otype_t::bool_v:
                return static_cast<double>(std::get<bool>(obj.v) != std::get<bool>(other.obj.v));
            case otype_t::int_v: {
                mpzw_t diff{};
                mpz_sub(diff.z, std::get<mpzw_t>(obj.v).z, std::get<mpzw_t>(other.obj.v).z);
                mpz_abs(diff.z, diff.z);
                return mpzs_compress_double(diff);
            }
            case otype_t::dec_v: {
                dec_t a = std::get<dec_t>(obj.v) - std::get<dec_t>(other.obj.v);
                mpfr_abs(a.mpfr_ptr(), a.mpfr_srcptr(), dec_t::get_default_rnd());
                dec_compress(a);
                return a.toDouble();
            }
            case otype_t::set: {
                if (exprs.size() > DIST_CHCK_SUBSET_MAX_SIZE || other.exprs.size() > DIST_CHCK_SUBSET_MAX_SIZE) {
                    return 0.01 + 0.99*std::sqrt(double_compress(std::abs(static_cast<double>(exprs.size()) - static_cast<double>(other.exprs.size())))); /* sqrt to push it higher towards being higher dist since they're different length */
                }
                for (const sptrexpr_t &expr : exprs) {
                }
                break;
            }
            default:
                return 1.0;
        }
    }

    std::string disp() const override {
        switch (obj.type) {
            case otype_t::bool_v:
                return (std::get<bool>(obj.v) ? "true" : "false");
            case otype_t::int_v: {
                char *s = nullptr;
                gmp_asprintf(&s, "%Zd", std::get<mpzw_t>(obj.v).z);
                std::string rs(s);
                free(s); /* NOLINT */
                return rs;
            }
            case otype_t::dec_v:
                return std::get<dec_t>(obj.v).toString();
            case otype_t::none:
                return "_noneobj";
            default:
                return "";
        }
    }
};


uptrexpr_t objexpr_t::copy_shallow() const {
    uptrexpr_t e = base_copy_shallow<objexpr_t>();
    uptrobjexpr_t oe = from_expr<objexpr_t>(e);
    if (obj.type == otype_t::int_v) {
        oe->obj.v = mpzw_t{};
        auto &nz = std::get<mpzw_t>(oe->obj.v);
        mpz_init_set(nz.z, std::get<mpzw_t>(obj.v).z);
    } else {
        oe->obj = obj;
    }
}

uptrexpr_t objexpr_t::copy() const {
    uptrexpr_t e = base_copy<objexpr_t>();
    uptrobjexpr_t oe = from_expr<objexpr_t>(e);
    oe->obj.type = obj.type;
    if (obj.type == otype_t::int_v) {
        oe->obj.v = mpzw_t{};
        auto &nz = std::get<mpzw_t>(oe->obj.v);
        mpz_init_set(nz.z, std::get<mpzw_t>(obj.v).z);
    } else {
        oe->obj = obj;
    }
    return to_expr(oe);
}

uptrobjexpr_t make_bool(bool v) {
    uptrobjexpr_t oe = std::make_unique<objexpr_t>();
    oe->obj.type = otype_t::bool_v;
    oe->obj.v = v;
    return oe;
}

/* uses mpz_init_set to copy */
uptrobjexpr_t make_mpzs_copy(mpzw_t i) {
    uptrobjexpr_t oe = std::make_unique<objexpr_t>();
    oe->obj.type = otype_t::int_v;
    oe->obj.v = mpzw_t{};
    auto &tz = std::get<mpzw_t>(oe->obj.v);
    mpz_init_set(tz.z, i.z);
    return oe;
}

/* steals mpz */
uptrobjexpr_t make_mpzs(mpzw_t i) {
    uptrobjexpr_t oe = std::make_unique<objexpr_t>();
    oe->obj.type = otype_t::int_v;
    oe->obj.v = mpzw_t{};
    std::get<mpzw_t>(oe->obj.v) = i;
    return oe;
}

uptrobjexpr_t make_dec(const dec_t &a) {
    uptrobjexpr_t oe = std::make_unique<objexpr_t>();
    oe->obj.type = otype_t::dec_v;
    oe->obj.v = a;
    return oe;
}

/* this is hard... for non lit objs */
uptrexpr_t objexpr_t::generate() const {
    switch (obj.type) {
        case otype_t::bool_v:
            return make_bool(get_random_int<bool>(false, true));
        case otype_t::int_v: {
            mpzw_t a{};
            mpz_init(a.z);
            mpzs_geometric_dist(a);
            return make_mpzs_copy(a);
        }
        case otype_t::dec_v: {
            dec_t a;
            mpfr_urandom(a.mpfr_ptr(), randstate, dec_t::get_default_rnd());
            return make_dec(a);
        }
        case otype_t::unresolvable:
        case otype_t::symbol:
        case otype_t::none:

        /* TODO: work on these !! hard :( */
        case otype_t::set:
        case otype_t::tuple:
            return copy();
    }
}

struct addexpr_t : virtual expr_t {

    addexpr_t() {
        type = etype_t::addexpr_t;
    }

    std::optional<uptrexpr_t> calculate() const override {
        std::variant<mpzw_t, dec_t> sum = mpzw_t{};
        mpz_init(std::get<mpzw_t>(sum).z);
        for (const sptrexpr_t &expr : exprs) {
            if (expr->type != etype_t::objexpr_t) {
                return {};
            }
            std::optional<uptrexpr_t> res = expr->calculate();
            if (res.has_value()) {
                uptrobjexpr_t ores = from_expr<objexpr_t>(res.value());
                if (sum.index() == 0) { /* is mpz */
                    if (ores->obj.type == otype_t::dec_v) {
                        dec_t tmp = dec_t(std::get<mpzw_t>(sum).z);
                        sum = tmp + std::get<dec_t>(ores->obj.v);
                    } else if (ores->obj.type == otype_t::int_v) {
                        mpz_add(std::get<mpzw_t>(sum).z, std::get<mpzw_t>(sum).z, std::get<mpzw_t>(ores->obj.v).z);
                    } else {
                        return {};
                    }
                } else if (sum.index() == 1) { /* is dec */
                    if (ores->obj.type == otype_t::dec_v) {
                        std::get<dec_t>(sum) += std::get<dec_t>(ores->obj.v);
                    } else if (ores->obj.type == otype_t::int_v) {
                        std::get<dec_t>(sum) += std::get<mpzw_t>(ores->obj.v).z;
                    } else {
                        return {};
                    }
                }
            } else {
                return {};
            }
        }
        uptrobjexpr_t r = std::make_unique<objexpr_t>();
        if (sum.index() == 0) { /* is mpz */
            r->obj.v = std::get<mpzw_t>(sum);
            r->obj.type = otype_t::int_v;
        } else if (sum.index() == 1) { /* is dec_t */
            r->obj.v = std::get<dec_t>(sum);
            r->obj.type = otype_t::dec_v;
        }
        return r;
    }

    std::string disp() const override {
        std::stringstream s;
        for (std::size_t i = 0; i < exprs.size() - 1; i++) {
            s << exprs[i]->disp();
            s << " + ";
        }
        if (!exprs.empty()) {
            s << exprs[exprs.size() - 1]->disp();
            return s.str();
        }
        return "_addexpr";
    }
};

DEF_ALL_PTRS(addexpr_t);

static constexpr std::size_t randgen_default_sample_size = 10;

double expr_t::dist_randgen(const expr_t &other, std::size_t sample_size = randgen_default_sample_size) const {
    sptrexpr_t e = copy(), o = other.copy();
    /* first we collect all objs */
    std::vector<sptrobjexpr_t> all_objs_e;
    /* TODO: !!!!! fix these pointer casts */
    /* e->walk_const([&all_objs_e](const sptrexpr_t &texpr) {
        if (texpr->type == etype_t::objexpr_t && texpr->exprs.empty()) {
            sptrobjexpr_t tobjexpr = from_expr_copy<objexpr_t>(texpr);
            if (tobjexpr->obj.name.has_value()) {
                all_objs_e.push_back(from_expr<objexpr_t>(texpr));
            }
        }
    }); */
    std::vector<sptrobjexpr_t> all_objs_o;
    /* TODO: !!!!! fix these pointer casts */
    /* o->walk_const([&all_objs_o](const sptrexpr_t &texpr) {
        if (texpr->type == etype_t::objexpr_t && texpr->exprs.empty()) {
            sptrobjexpr_t tobjexpr = std::static_pointer_cast<objexpr_t>(texpr);
            if (tobjexpr->obj.name.has_value()) {
                all_objs_o.push_back(tobjexpr);
            }
        }
    }); */

    auto objnamecomp = [](const sptrobjexpr_t &a, const sptrobjexpr_t &b) {
        return a->obj.name.value() < b->obj.name.value();
    };
    std::sort(all_objs_e.begin(), all_objs_e.end(), objnamecomp);
    std::sort(all_objs_o.begin(), all_objs_o.end(), objnamecomp);

    vec_t dvec;
    std::vector<std::pair<std::optional<sptrobjexpr_t>, std::optional<sptrobjexpr_t>>> all_objs;
    all_objs.reserve(all_objs_e.size() + all_objs_o.size());
    dvec.v.resize(all_objs_e.size() + all_objs_o.size());
    std::uint32_t i = 0, j = 0;
    while (i < all_objs_e.size() && j < all_objs_o.size()) {
        sptrobjexpr_t &teo = all_objs_e[i++];
        sptrobjexpr_t &too = all_objs_o[j++];
        if (teo == too) {
            all_objs.emplace_back(teo, too);
        } else {
            if (teo->obj.name.value() < too->obj.name.value()) {
                all_objs.emplace_back(teo, std::optional<sptrobjexpr_t>{});
                j--;
            } else {
                all_objs.emplace_back(std::optional<sptrobjexpr_t>{}, too);
                i--;
            }
        }
    }
    if (i < all_objs_e.size()) {
        for (; i < all_objs_e.size(); i++) {
            all_objs.emplace_back(all_objs_e[i], std::optional<sptrobjexpr_t>{});
        }
    } else if (j < all_objs_o.size()) {
        for (; j < all_objs_o.size(); j++) {
            all_objs.emplace_back(std::optional<sptrobjexpr_t>{}, all_objs_o[j]);
        }
    }

    double dsum = 0;
    std::size_t samples_taken = 0;
    for (std::size_t j = 0; j < sample_size; j++) {
        for (auto &[teo, too] : all_objs) {
            if (teo.has_value()) {
                uptrexpr_t gen = teo.value()->generate();
                std::unique_ptr<objexpr_t> uo = from_expr<objexpr_t>(gen);
                teo.value()->obj = uo->obj;
                teo.value()->signal_dirty();
            }
            if (too.has_value()) {
                uptrexpr_t gen = too.value()->generate();
                std::unique_ptr<objexpr_t> uo = from_expr<objexpr_t>(gen);
                too.value()->obj = uo->obj;
                too.value()->signal_dirty();
            }
        }
        std::optional<uptrexpr_t> re = e->calculate();
        if (re.has_value()) {
            std::optional<uptrexpr_t> ro = o->calculate();
            if (ro.has_value()) {
                uptrobjexpr_t reo = from_expr<objexpr_t>(re.value());
                uptrobjexpr_t roo = from_expr<objexpr_t>(ro.value());
                if (reo->obj.type != roo->obj.type) {
                    dsum += 1.0;
                } else {
                    dsum += reo->obj_dist(*roo);
                }
                samples_taken++;
            }
        }
    }

    if (samples_taken == 0) {
        return 1.0;
    }

    return dsum / static_cast<double>(samples_taken);
}

double expr_t::dist_match(const expr_t &other) const {
    std::vector<std::vector<etype_t>> stacks, ostacks;
    std::vector<etype_t> tmp_cur, otmp_cur;
    get_deepest_type_stacks(stacks, tmp_cur);
    other.get_deepest_type_stacks(ostacks, otmp_cur);
    double simsum = 0.0;
    std::size_t simcnt = stacks.size() * ostacks.size();
    for (const std::vector<etype_t> &tstack : stacks) {
        for (const std::vector<etype_t> &ostack : ostacks) {
            simsum += vec_similarity(tstack, ostack);
        }
    }
    double avg = 1.0;
    if (simcnt > 0) {
        avg = simsum/static_cast<double>(simcnt);
    }
    return avg;
}

double expr_t::dist(const expr_t &other, std::size_t sample_size = randgen_default_sample_size) const {
    double m = dist_match(other);
    double rd = dist_randgen(other, sample_size);
    return std::sqrt(m * m + rd * rd);
}


int main() {
    gmp_randinit_default(randstate);

    // void *dlhandle = dlopen(WORK_SO_FILENAME, RTLD_LAZY);
    // expr_work = reinterpret_cast<decltype(expr_work)>(dlsym(dlhandle, "expr_work")); /* NOLINT */
    // if (expr_work == nullptr) {
    //     ERR_EXIT(1, "could not resolve expr_work from shared object " WORK_SO_FILENAME);
    // }
    // dlclose(dlhandle);

    mpzw_t i{};
    mpz_init_set_ui(i.z, 3);
    sptrobjexpr_t three = make_mpzs_copy(i);
    mpz_set_ui(i.z, 2);
    sptrobjexpr_t two = make_mpzs(i);
    sptraddexpr_t a = std::make_unique<addexpr_t>();
    expr_add_expr(a, two);
    expr_add_expr(a, three);
    uptrexpr_t res = a->load(true);
    uptrobjexpr_t ores = from_expr<objexpr_t>(res);
    std::cout << a->disp() << " loaded to " << ores->disp() << std::endl;

    gmp_randclear(randstate);
}
