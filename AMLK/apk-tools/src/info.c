/* info.c - Alpine Package Keeper (APK)
 *
 * Copyright (C) 2005-2009 Natanael Copa <n@tanael.org>
 * Copyright (C) 2009 Timo Teräs <timo.teras@iki.fi>
 * All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 as published
 * by the Free Software Foundation. See http://www.gnu.org/ for details.
 */

#include <stdio.h>
#include <unistd.h>
#include "apk_defines.h"
#include "apk_applet.h"
#include "apk_package.h"
#include "apk_database.h"
#include "apk_state.h"
#include "apk_print.h"

struct info_ctx {
	int (*action)(struct info_ctx *ctx, struct apk_database *db,
		      int argc, char **argv);
	int subaction_mask;
};

/* These need to stay in sync with the function pointer array in
 * info_subaction() */
#define APK_INFO_DESC		0x01
#define APK_INFO_URL		0x02
#define APK_INFO_SIZE		0x04
#define APK_INFO_DEPENDS	0x08
#define APK_INFO_RDEPENDS	0x10
#define APK_INFO_CONTENTS	0x20
#define APK_INFO_TRIGGERS	0x40

static void verbose_print_pkg(struct apk_package *pkg, int minimal_verbosity)
{
	int verbosity = apk_verbosity;
	if (verbosity < minimal_verbosity)
		verbosity = minimal_verbosity;

	if (pkg == NULL || verbosity < 1)
		return;

	printf("%s", pkg->name->name);
	if (apk_verbosity > 1)
		printf("-%s", pkg->version);
	if (apk_verbosity > 2)
		printf(" - %s", pkg->description);
	printf("\n");
}


static int info_list(struct info_ctx *ctx, struct apk_database *db,
		     int argc, char **argv)
{
	struct apk_installed_package *ipkg;

	list_for_each_entry(ipkg, &db->installed.packages, installed_pkgs_list)
		verbose_print_pkg(ipkg->pkg, 1);
	return 0;
}

static int info_exists(struct info_ctx *ctx, struct apk_database *db,
		       int argc, char **argv)
{
	struct apk_name *name;
	struct apk_package *pkg = NULL;
	struct apk_dependency dep;
	int r, i, j, ok = 0;

	for (i = 0; i < argc; i++) {
		r = apk_dep_from_blob(&dep, db, APK_BLOB_STR(argv[i]));
		if (r != 0)
			continue;

		name = dep.name;
		for (j = 0; j < name->pkgs->num; j++) {
			pkg = name->pkgs->item[j];
			if (pkg->ipkg != NULL)
				break;
		}
		if (j >= name->pkgs->num)
			continue;

		if (!(apk_version_compare(pkg->version, dep.version)
		      & dep.result_mask))
			continue;

		verbose_print_pkg(pkg, 0);
		ok++;
	}

	return argc - ok;
}

static int info_who_owns(struct info_ctx *ctx, struct apk_database *db,
			 int argc, char **argv)
{
	struct apk_package *pkg;
	struct apk_dependency_array *deps;
	struct apk_dependency dep;
	int i, r=0;

	apk_dependency_array_init(&deps);
	for (i = 0; i < argc; i++) {
		pkg = apk_db_get_file_owner(db, APK_BLOB_STR(argv[i]));
		if (pkg == NULL) {
			apk_error("%s: Could not find owner package", argv[i]);
			r++;
			continue;
		}

		if (apk_verbosity < 1) {
			dep = (struct apk_dependency) {
				.name = pkg->name,
				.result_mask = APK_DEPMASK_REQUIRE,
			};
			apk_deps_add(&deps, &dep);
		} else {
			printf("%s is owned by %s-%s\n", argv[i],
			       pkg->name->name, pkg->version);
		}
	}
	if (apk_verbosity < 1 && deps->num != 0) {
		struct apk_ostream *os;

		os = apk_ostream_to_fd(STDOUT_FILENO);
		apk_deps_write(deps, os);
		os->write(os, "\n", 1);
		os->close(os);
	}
	apk_dependency_array_free(&deps);

	return r;
}

static void info_print_description(struct apk_package *pkg)
{
	if (apk_verbosity > 1)
		printf("%s: %s", pkg->name->name, pkg->description);
	else
		printf("%s-%s description:\n%s\n", pkg->name->name,
		       pkg->version, pkg->description);
}

static void info_print_url(struct apk_package *pkg)
{
	if (apk_verbosity > 1)
		printf("%s: %s", pkg->name->name, pkg->url);
	else
		printf("%s-%s webpage:\n%s\n", pkg->name->name, pkg->version,
		       pkg->url);
}

static void info_print_size(struct apk_package *pkg)
{
	if (apk_verbosity > 1)
		printf("%s: %zu", pkg->name->name, pkg->installed_size);
	else
		printf("%s-%s installed size:\n%zu\n", pkg->name->name, pkg->version,
		       pkg->installed_size);
}

static void info_print_depends(struct apk_package *pkg)
{
	int i;
	char *separator = apk_verbosity > 1 ? " " : "\n";
	if (apk_verbosity == 1)
		printf("%s-%s depends on:\n", pkg->name->name, pkg->version);
	if (apk_verbosity > 1)
		printf("%s: ", pkg->name->name);
	for (i = 0; i < pkg->depends->num; i++)
		printf("%s%s", pkg->depends->item[i].name->name, separator);
}

static void info_print_required_by(struct apk_package *pkg)
{
	int i, j, k;
	char *separator = apk_verbosity > 1 ? " " : "\n";

	if (apk_verbosity == 1)
		printf("%s-%s is required by:\n", pkg->name->name, pkg->version);
	if (apk_verbosity > 1)
		printf("%s: ", pkg->name->name);
	for (i = 0; i < pkg->name->rdepends->num; i++) {
		struct apk_name *name0 = pkg->name->rdepends->item[i];

		/* Check only the package that is installed, and that
		 * it actually has this package as dependency. */
		for (j = 0; j < name0->pkgs->num; j++) {
			struct apk_package *pkg0 = name0->pkgs->item[j];

			if (pkg0->ipkg == NULL)
				continue;
			for (k = 0; k < pkg0->depends->num; k++) {
				if (pkg0->depends->item[k].name != pkg->name)
					continue;
				printf("%s-%s%s", pkg0->name->name,
				       pkg0->version, separator);
				break;
			}
		}
	}
}

static void info_print_contents(struct apk_package *pkg)
{
	struct apk_installed_package *ipkg = pkg->ipkg;
	struct apk_db_dir_instance *diri;
	struct apk_db_file *file;
	struct hlist_node *dc, *dn, *fc, *fn;

	if (apk_verbosity == 1)
		printf("%s-%s contains:\n", pkg->name->name, pkg->version);

	hlist_for_each_entry_safe(diri, dc, dn, &ipkg->owned_dirs,
				  pkg_dirs_list) {
		hlist_for_each_entry_safe(file, fc, fn, &diri->owned_files,
					  diri_files_list) {
			if (apk_verbosity > 1)
				printf("%s: ", pkg->name->name);
			printf("%s/%s\n", diri->dir->name, file->name);
		}
	}
}

static void info_print_triggers(struct apk_package *pkg)
{
	struct apk_installed_package *ipkg = pkg->ipkg;
	int i;

	if (apk_verbosity == 1)
		printf("%s-%s triggers:\n", pkg->name->name, pkg->version);

	for (i = 0; i < ipkg->triggers->num; i++) {
		if (apk_verbosity > 1)
			printf("%s: trigger ", pkg->name->name);
		printf("%s\n", ipkg->triggers->item[i]);
	}
}

static void info_subaction(struct info_ctx *ctx, struct apk_package *pkg)
{
	typedef void (*subaction_t)(struct apk_package *);
	static subaction_t subactions[] = {
		info_print_description,
		info_print_url,
		info_print_size,
		info_print_depends,
		info_print_required_by,
		info_print_contents,
		info_print_triggers,
	};
	const int requireipkg =
		APK_INFO_CONTENTS | APK_INFO_TRIGGERS | APK_INFO_RDEPENDS;
	int i;

	for (i = 0; i < ARRAY_SIZE(subactions); i++) {
		if (!(BIT(i) & ctx->subaction_mask))
			continue;

		if (pkg->ipkg == NULL && (BIT(i) & requireipkg))
			continue;

		subactions[i](pkg);
		puts("");
	}
}

static int info_package(struct info_ctx *ctx, struct apk_database *db,
			int argc, char **argv)
{
	struct apk_name *name;
	int i, j;

	for (i = 0; i < argc; i++) {
		name = apk_db_query_name(db, APK_BLOB_STR(argv[i]));
		if (name == NULL || name->pkgs->num == 0) {
			apk_error("Not found: %s", argv[i]);
			return 1;
		}
		for (j = 0; j < name->pkgs->num; j++)
			info_subaction(ctx, name->pkgs->item[j]);
	}
	return 0;
}

static int info_parse(void *ctx, struct apk_db_options *dbopts,
		      int optch, int optindex, const char *optarg)
{
	struct info_ctx *ictx = (struct info_ctx *) ctx;

	ictx->action = info_package;
	switch (optch) {
	case 'e':
		ictx->action = info_exists;
		break;
	case 'W':
		ictx->action = info_who_owns;
		break;
	case 'w':
		ictx->subaction_mask |= APK_INFO_URL;
		break;
	case 'R':
		ictx->subaction_mask |= APK_INFO_DEPENDS;
		break;
	case 'r':
		ictx->subaction_mask |= APK_INFO_RDEPENDS;
		break;
	case 's':
		ictx->subaction_mask |= APK_INFO_SIZE;
		break;
	case 'd':
		ictx->subaction_mask |= APK_INFO_DESC;
		break;
	case 'L':
		ictx->subaction_mask |= APK_INFO_CONTENTS;
		break;
	case 't':
		ictx->subaction_mask |= APK_INFO_TRIGGERS;
		break;
	case 'a':
		ictx->subaction_mask = 0xffffffff;
		break;
	default:
		return -1;
	}
	return 0;
}

static int info_main(void *ctx, struct apk_database *db, int argc, char **argv)
{
	struct info_ctx *ictx = (struct info_ctx *) ctx;

	if (ictx->action != NULL)
		return ictx->action(ictx, db, argc, argv);

	return info_list(ictx, db, argc, argv);
}

static struct apk_option info_options[] = {
	{ 'L', "contents",	"List contents of the PACKAGE" },
	{ 'e', "installed",	"Check if PACKAGE is installed" },
	{ 'W', "who-owns",	"Print the package owning the specified file" },
	{ 'R', "depends",	"List packages that the PACKAGE depends on" },
	{ 'r', "rdepends",	"List all packages depending on PACKAGE" },
	{ 'w', "webpage",	"Show URL for more information about PACKAGE" },
	{ 's', "size",		"Show installed size of PACKAGE" },
	{ 'd', "description",	"Print description for PACKAGE" },
	{ 't', "triggers",	"Print active triggers of PACKAGE" },
	{ 'a', "all",		"Print all information about PACKAGE" },
};

static struct apk_applet apk_info = {
	.name = "info",
	.help = "Give detailed information about PACKAGEs or repositores.",
	.arguments = "PACKAGE...",
	.open_flags = APK_OPENF_READ,
	.context_size = sizeof(struct info_ctx),
	.num_options = ARRAY_SIZE(info_options),
	.options = info_options,
	.parse = info_parse,
	.main = info_main,
};

APK_DEFINE_APPLET(apk_info);

